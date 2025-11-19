import argparse
import transformers
from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Dict


def get_args_parser():
    parser = argparse.ArgumentParser(description="args for training", add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--mm_projector_lr", type=float, default=None)

    parser.add_argument("--version", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

    parser.add_argument("--log_base", type=str, default="logs_new")
    parser.add_argument("--exp_name", type=str, default="m2t")

    return parser.parse_args()


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # need output_dir
    cache_dir: Optional[str] = field(default="/home/sxu/.llama/checkpoints/Llama-3.1-8B-Instruct")
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    output_dir: Optional[str] = field(default="./logs/")
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    precision: str = field(default="bf16") # choices=["fp32", "bf16", "fp16"]
    log_base: str = field(default="logs")
    exp_name: str = field(default="m2t")

    use_v2: bool = False

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct")
    version: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct")
    # model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-3B-Instruct")
    # version: Optional[str] = field(default="meta-llama/Llama-3.2-3B-Instruct")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="mlp")
    mm_vision_select_layer: Optional[str] = field(default="") # default to last layer
    mm_vision_select_feature: Optional[str] = field(default="")
    pretrain_mm: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear") # linear or mlp2x_gelu
    mm_use_start_end: bool = field(default=False)
    mm_use_patch_tokens: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default="")
    mm_hidden_size: int = field(default=1024)

    # for motion_process
    motion_dim: int = field(default=66*16) # motion_dim: 22*3*window_size; here window_size is 16
    mlp_hidden: int = field(default=1024) 

    # conv type
    conv_type: str = field(default="llava_llama_3") # llava_3, plain, llava_llama_3

@dataclass
class DataArguments:
    data_root: str = field(default="/home/sxu/HumanML3D/HumanML3D")
    data_path: str = field(default="./datasets/stage1.json") # json file include converstaion
    motion_folder: str = field(default="/home/sxu/HumanML3D/HumanML3D/new_joints")
    split: str = field(default="train")
    window_size: int = field(default=16)
    lazy_preprocess: bool = False
