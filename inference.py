import argparse
import os
import torch
import numpy as np
from model.m2t_llama import M2TLlamaForCausalLM, M2TConfig
import json
import transformers
import copy
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from data.datasets import rotate_y_up_to_z_up
from utils.conversation import default_conversation, conv_templates
from utils.utils import disable_torch_init, DEFAULT_MOTION_TOKEN, MOTION_TOKEN_INDEX
from utils.mm_utils import tokenizer_motion_token
from peft import PeftModel
from data.generate_json import time_convert


access_token = "xxxxxx" # your huggingface access token here

def generate_descriptsion(args):
    disable_torch_init()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    config = AutoConfig.from_pretrained(args.model_path)

    if args.motion_dim is not None:
        print(config)
        for k, v in vars(args).items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif k in ['motion_dim', 'mlp_hidden']:
                setattr(config, k, v)

    model = M2TLlamaForCausalLM.from_pretrained(args.model_path, config=config, device_map="auto")

    model.eval()
    print(model)

    system_message = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."

    datas = json.load(open(args.data, "r"))
    input_ids = []
    outputs = []
    for data in tqdm(datas):
        results = {}
        if "motion" in data:
            motion = np.load(data["motion"])
            if motion.shape[0] < 20: continue
            duration = motion.shape[0] / 20
            duration = time_convert(duration)
            motion = torch.tensor(motion, dtype=torch.float32)
            motion = rotate_y_up_to_z_up(motion).to(model.device)
            if motion.ndim == 3:
                if motion.shape[1] == 24:
                    motion = motion[:, :22, :]
            human_input = f"<motion>\n\nGiven a complex human motion sequence of duration {duration} which includes several actions, describe these actions in the motion with natural language according to the movement of human. \n" \
                            "The description of each action should be in the format 'mm:ss:ms - text'. \n" \
                            "Here is an example: 00:00:00 - moves in a curve to the right side, 00:05:09 - doing a left foot squat \n"
            
            if DEFAULT_MOTION_TOKEN in human_input:
                human_input = human_input.replace(DEFAULT_MOTION_TOKEN, '').strip()
                human_input = DEFAULT_MOTION_TOKEN + "\n" + human_input
                human_input = human_input.strip()
        else:
            print("oooops! sry you must give the motion!!")


        conv_template = "llama_3"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.tokenizer = tokenizer
        conv.append_message(conv.roles[0], human_input)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_motion_token(prompt_question, tokenizer, MOTION_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        output = model.generate(input_ids, [motion], attention_mask=attention_mask, do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)

        output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        results["id"] = data["id"]
        results["generated"] = output
        print("output: ", output)
        # ['a person walks forward at a normal pace.']
        outputs.append(results)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="logs/stage2")
    parser.add_argument("--model_base", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", type=str, default="datasets/test.json")
    parser.add_argument("--output", type=str, default="datasets/results/test.json")
    parser.add_argument("--motion_dim", type=int, default=None) # 1152, 1056
    parser.add_argument("--mlp_hidden", type=int, default=1024)

    args = parser.parse_args()
    generate_descriptsion(args)
    