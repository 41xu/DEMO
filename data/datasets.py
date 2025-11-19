import torch
import torch.nn as nn
import transformers
import json
import os
import copy
import numpy as np
from typing import Dict, Optional, List, Sequence
from dataclasses import dataclass, field
from torch.utils.data import Dataset
# import sys
# sys.path.append(".")
# sys.path.append("..")
from utils.utils import IGNORE_INDEX, MOTION_TOKEN_INDEX, DEFAULT_MOTION_TOKEN, DEFAULT_MOTION_START_TOKEN, DEFAULT_MOTION_END_TOKEN
from utils import conversation as conversation_lib
from utils.mm_utils import tokenizer_motion_token

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def rotate_y_up_to_z_up(data):
    """
    data: (..., 3) shape: [N, 3] or [T, J, 3]
    return: rotated data
    """
    rot = np.array([
        [1, 0,  0],
        [0, 0,  1],
        [0, -1, 0]
    ])
    original_shape = data.shape
    reshaped = data.reshape(-1, 3)
    rotated = reshaped @ rot.T
    return rotated.reshape(original_shape)


def preprocess_multimodal(sources, args):
    is_multimodal = args.is_multimodal # False
    if not is_multimodal:
        return sources
    """sources: original conversation list, but with 2 layer list, therefore they they use for and for loop

    sources:  [[{'from': 'human', 'value': 'Build a narrative description that matches the stated series of human motion cues.'}, 
                {'from': 'gpt', 'value': 'a person bows slightly briefly.'}]]
    """
    for source in sources:
        for sentence in source:
            if DEFAULT_MOTION_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_MOTION_TOKEN, '').strip()
                sentence["value"] = DEFAULT_MOTION_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_MOTION_TOKEN, '<Motion>' + DEFAULT_MOTION_TOKEN + '</Motion>')
            replace_token = DEFAULT_MOTION_TOKEN
            if args.mm_use_start_end:
                replace_token = DEFAULT_MOTION_START_TOKEN + replace_token + DEFAULT_MOTION_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_MOTION_TOKEN, replace_token)
    return sources

# FIXME
def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_motion: bool = False,
    max_len=2048,
    # system_message: str = "You are a helpful huamn motion understanding assistant. You are able to understand the human motions that the user provides, and assist the user with a variety of tasks using natural language." \
    # "Given a complex human motion sequence which includes several actions, you are able to distinguish different actions in the motion and describe the actions in the motion with natural language according to the movements." \
    # " The description of each action should be in the format 'mm:ss:ms - text'. Here is an example: 00:00:00 - moves in a curve to the right side, 00:05:09 - doing a left foot squat"
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_motion:
        tokenizer.add_tokens(["<motion>"], special_tokens=True)
    motion_token_index = tokenizer.convert_tokens_to_ids("<motion>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        """source here is just one-loop list. thanks...
        [{'from': 'human', 'value': '<motion>\nPlease explain the movement being represented by  using text.'}, 
        {'from': 'gpt', 'value': 'a person jumps sideways to the left'}]"""
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == motion_token_index:
                input_id[idx] = MOTION_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    # process prompt got tokens and add image_token mask for final loss calculation
    conversations = []
    """
    sources in process plain:  [{'from': 'human', 'value': '<motion>\nDescribe the actions by using text.'}, 
    {'from': 'gpt', 'value': 'a person bends their knees slightly jumps forward and stops.'}]
    """
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_MOTION_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_MOTION_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
        # this convesation is <motion>+gpt_description+\n\n
    # tokenize conversations
    input_ids = [tokenizer_motion_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_motion_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess(sources, tokenizer, has_motion=True):
    """
    Given a list of sources, each is a conversation list, transform:
    1. Add signal '###' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concated conversation; <<< this part we need to embed the motions?
    4. Make a deepcopy as the target, Mask human words with IGNORE_INDEX
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama3(sources, tokenizer, has_motion=has_motion)
    
    # conversations = []
    # for source in sources:
    #     header = f"{conversation_lib.default_conversation.system}\n\n"
    #     conversation = _add_speaker_and_signal(header, source)
    #     conversations.append(conversation)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, args, tokenizer):
        super(LazySupervisedDataset, self).__init__()
        data_path = args.data_path 
        list_data_dict = json.load(open(data_path, "r"))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict # the json file
        self.args = args

    def __len__(self):
        return len(self.list_data_dict)
    """
        {
        "id": "000004",
        "motion": "/home/sxu/HumanML3D/HumanML3D/new_joints/000004.npy",
        "conversations": [
            {
                "from": "human",
                "value": "Explain the motion using language.\n<motion>"
            },
            {
                "from": "gpt",
                "value": "humanoid is acting like a chicken with their arms pointed up and their hands on their chest area, they also were pecking at something"
            }
        ]
    },
    """
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # i think here the motion_token length depends on the len(motions)
            motion_tokens = 2000 if 'motion' in sample else 0 # TODO: here the max_motion_token_lens still need to be tuned. 200 is temporal depends on the data json process scripts
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + motion_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'motion' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        sources = self.list_data_dict[i]

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # here is just for tokenize?
        if 'motion' in sources[0]:
            motion_file = self.list_data_dict[i]['motion']
            # no processor
            motion = np.load(motion_file)
            motion = torch.tensor(motion, dtype=torch.float32)
            # motion = rotate_y_up_to_z_up(motion)
            if motion.ndim == 3:
                if motion.shape[1] == 24:
                    motion = motion[:, :22, :]
            # do no preprocess, directly use the motion data.
            # sources = preprocess_multimodal(copy.deepcopy(sources["conversations"]), self.args)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            # sources = copy.deepcopy(sources["conversations"])

        data_dict = preprocess(sources, self.tokenizer, has_motion=('motion' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
    
        # # image exist in the data
        # if 'motion' in self.list_data_dict[i]:
        #     data_dict['motion'] = motion
        # # i think this branch is only for motiongpt, they have motion encoder
        # elif self.data_args.is_multimodal: # not used, no is_multimodal
        #     # image does not exist in the data, but the model is multimodal
        #     # crop_size = self.data_args.image_processor.crop_size
        #     # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        #     data_dict['motion'] = torch.zeros(200, 66)
        data_dict['motion'] = motion 
        """
        data_dict: {motion, input_ids, labels}
        """
        return data_dict

# TBD
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0 # fuck! without this zero-padding the code will have errors!
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'motion' in instances[0]:
            motions = [instance['motion'] for instance in instances]
            # motions = [rotate_y_up_to_z_up(motion) for motion in motions]
            motion_type = ["joint" for _ in range(len(motions))]
            if motions[0].ndim == 3 and motions[0].shape[-1] * motions[0].shape[-2] == 66:
                batch["motion_type"] = motion_type
            elif motions[0].ndim == 2 and motions[0].shape[-1] == 263:
                motion_type = ["humanml" for _ in range(len(motions))]
                batch["motion_type"] = motion_type
            
            if all(x is not None and x.shape == motions[0].shape for x in motions):
                batch['motion'] = torch.stack(motions)
            else:
                batch['motion'] = motions

        return batch


def make_supervised_data_module(args, tokenizer):
    """Make dataset and collator for supervised fine-tuning"""
    train_dataset = LazySupervisedDataset(args, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    return dict(train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator)


if __name__ == '__main__':
    from options.option import get_args_parser, TrainingArguments, ModelArguments, DataArguments
    from transformers import AutoTokenizer
    import torch
    import datasets
    from torch.utils.data import DataLoader
    from transformers.trainer_pt_utils import LengthGroupedSampler
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.is_multimodal = True
    data_args.mm_use_start_end = False

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(data_args, tokenizer)
    dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    dataloader = DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        print(batch)
    # 🖕🏼 test, works well.