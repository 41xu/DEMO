import argparse
import os
import sys
import time
import pathlib
import deepspeed
import numpy as np
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from options.option import get_args_parser, TrainingArguments, ModelArguments, DataArguments
from utils.utils import DEFAULT_MOTION_START_TOKEN, DEFAULT_MOTION_END_TOKEN
from data.datasets import make_supervised_data_module
from model.m2t_llama import M2TLlamaForCausalLM, M2TConfig
from utils.helper import *
from utils import conversation as conversation_lib
from model.m2t_trainer import M2TTrainer
# from inference import generate_descriptsion


access_token = "xxxxxx" # your huggingface access token here

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def main(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)
    if training_args.lora_enable:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                                            cache_dir=training_args.cache_dir, 
                                                            token=access_token,
                                                            model_max_length=training_args.model_max_length,
                                                            padding_side="right",
                                                            use_fast=False)

        # according to model_args.version. pretrain stage: plain
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if not tokenizer.pad_token_id:
        tokenizer.unk_token_id = tokenizer.pad_token_id
    tokenizer.add_tokens("<motion>", special_tokens=True)
    
    print("tokenizer length: ", len(tokenizer))
    local_rank = training_args.local_rank
    # local_rank = args.local_rank
    training_args.log_dir = os.path.join(training_args.log_base, training_args.exp_name)
    if local_rank == 0: 
        os.makedirs(training_args.log_dir, exist_ok=True)
        writer = SummaryWriter(training_args.log_dir)
    else:
        writer = None
    
    compute_dtype = torch.float16 if training_args.precision == "fp16" else (torch.bfloat16 if training_args.precision == "bf16" else torch.float32)
    print("comput_dtype: ", compute_dtype)
    model = M2TLlamaForCausalLM.from_pretrained(
        model_args.version, # llama path
        cache_dir=training_args.cache_dir,
        token=access_token,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16 if training_args.precision=="bf16" else None)

    model.resize_token_embeddings(len(tokenizer)) 
    
    rank0_print(model)
    model.config.use_cache = False
    if model_args.freeze_backbone: # pretrain stage
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing: # True?
        if hasattr(model, "enable_input_require_grads"):
            print("oh wow")
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.precision == "bf16":
            model.to(torch.bfloat16)
        if training_args.precision == "fp16":
            model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    else:
        if training_args.precision == "bf16":
            model.to(torch.bfloat16)
        if training_args.precision == "fp16":
            model.to(torch.float16)

    data_args.mm_use_start_end = model_args.mm_use_start_end

    # if model_args.mm_use_start_end: # False
    #     tokenizer.add_tokens([DEFAULT_MOTION_START_TOKEN, DEFAULT_MOTION_END_TOKEN], special_tokens=True)
    
    if model_args.conv_type in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.conv_type]
    else: # default
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    # NOTE: here vision_tower should be none, but for llava they introduce encoder and proj_layer in vision_tower. 
    # NOTE: therefore, here we keep the vision_tower part, but only for motion_project_layer. no encoder.
    print("model-args: ", model_args)
    print("----------------------------------------")
    if model_args.vision_tower != '':
        # HERE vision tower and mlp_projector is loaded in the finetuning stage.
        model.get_model().initialize_motion_modules(model_args, fsdp=training_args.fsdp)
        vision_tower = model.get_vision_tower().to(dtype=compute_dtype)

        data_args.is_multimodal = True
        # no image_processor so no data_args.image_processor = vision_tower.image_processor

        model.config.tokenizer_padding_size = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

        if model_args.tune_mm_mlp_adapter: # Stage1, True, always true. also need to train vision_tower
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            model.get_model().mm_projector.to(dtype=compute_dtype)

        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        
        model.config.mm_use_start_end = model_args.mm_use_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    else: # just for motiongpt motion_tokens as input
        model.get_model().vision_tower = nn.Identity()
        model.get_model().mm_projector = nn.Identity()
        print("none branch")

    data_module = make_supervised_data_module(data_args, tokenizer)
    # print(model.get_model().get_vision_tower())
    trainer = M2TTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    rank0_print("now we in the training(trainer)")
    if list(pathlib.Path(training_args.log_dir).glob("checkpoint=*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable: # True
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model = model.merge_and_unload()
            model.resize_token_embeddings(len(tokenizer)) 

            model.config.save_pretrained(training_args.log_dir)
            model.save_pretrained(training_args.log_dir)
            tokenizer.save_pretrained(training_args.log_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.log_dir)
        tokenizer.save_pretrained(training_args.log_dir)

if __name__ == '__main__':
    main()