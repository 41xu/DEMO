import torch
from utils.utils import MOTION_TOKEN_INDEX


def tokenizer_motion_token(prompt, tokenizer, motion_token_index=MOTION_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<motion>")]

    def insert_separator(X, sep): # instert sep in X. 
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    # if prompt starts with bos_token_id, skip. and bos_token_id is added into input_ids
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    # insert image_token_index into prompt_chunks and skip bos_token
    for x in insert_separator(prompt_chunks, [motion_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

