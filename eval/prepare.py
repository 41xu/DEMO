import numpy as np
import json
import re
import os
from tqdm import tqdm
import torch.nn as nn
import clip
import torch
import argparse



def convert_time_to_seconds(time_str):
    mm, ss, hs = map(int, time_str.split(':'))
    total_seconds = mm * 60 + ss + hs / 1000.0
    return total_seconds

class Clip_Encoder(nn.Module):
    def __init__(self, clip_version=None, clip_dim=512, latent_dim=512, dataset= 'babel', **kargs):
        super().__init__()
        self.clip_dim = clip_dim
        self.latent_dim = int(latent_dim) 
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        print('EMBED TEXT')
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)
        if torch.cuda.is_available():
            self.clip_model = self.clip_model.to('cuda')
        else:
            print("CUDA is not available. The model will remain on the CPU.")
        self.dataset = dataset

    def load_and_freeze_clip(self,clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        # device = next(self.parameters()).device #cpu 
        device = 'cuda'
        # print(device)
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    def forward(self, text, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        enc_text = self.encode_text(text)
        # emb = self.embed_text(enc_text)
        return enc_text
    


def main():
    pattern = r"(\d{2}:\d{2}:\d{1,3})\s*-\s*([^,]+(?:,[^,\d{2}]+)*)"
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str)
    parser.add_argument("--ref_file", type=str, default="../datasets/test.json")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--output_gt_file", type=str, default="data/gt.json")
    parser.add_argument("--comp_path", type=str, help="path to comp dataset, used to obtain duration info")
    args = parser.parse_args()

    pred_file = args.pred_file
    gt_file = args.ref_file
    
    os.makedirs("data", exist_ok=True)
    output_file = args.output_file if args.output_file is not None else "data/" + pred_file.split("/")[-1]

    res_gt = {}
    with open(gt_file, 'r') as f:
        gtdata = json.load(f)
    for data in gtdata:
        timestamps = []
        setences = []
        dur = np.load(data["motion"]).shape[0] / 20
        caps = data["gt"]
        time_tmp = []
        matches = re.findall(pattern, caps)
        for x in matches:
            t, s = x[0].strip(), x[1].strip()
            if s[-1] == ',':
                s = s[:-1]
            time_tmp.append(t)
            setences.append(s)
        
        time_tmp = [convert_time_to_seconds(t) for t in time_tmp]
        for i in range(len(time_tmp)-1):
            timestamps.append([time_tmp[i], time_tmp[i+1]])
        timestamps.append([time_tmp[-1], dur])
        res_gt[data["id"]] = {
            "timestamps": timestamps,
            "sentences": setences
        }
    with open(args.output_gt_file, "w") as f:
        json.dump(res_gt, f, indent=4)

    res_pred = {}
    res = {}
    with open(pred_file, 'r') as f:
        preddata = json.load(f)
    for data in preddata:
        res_list = []
        timestamps = []
        time_tmp = []
        setences = []
        dur = np.load(os.path.join(args.comp_path, f"{data['id']}.npy")).shape[0] / 20
        caps = data["generated"]
        matches = re.findall(pattern, caps)
        for x in matches:
            t, s = x[0].strip(), x[1].strip()
            if s[-1] == ',':
                s = s[:-1]
            time_tmp.append(t)
            setences.append(s)
        time_tmp = [convert_time_to_seconds(t) for t in time_tmp]
        if len(time_tmp) == 0:
            continue
        for i in range(len(time_tmp) - 1):
            timestamps.append([time_tmp[i], time_tmp[i + 1]])
        print(len(time_tmp), time_tmp)
        timestamps.append([time_tmp[-1], dur])
        assert len(timestamps) == len(setences)
        for i in range(len(timestamps)):
            res_list.append({"timestamp": timestamps[i], "sentence": setences[i]})
        res[data["id"]] = res_list
    res_pred["results"] = res
        # res_pred["results"] = dict
        # dict: "id":[{"timestamp": [], "sentence": ""}, ...]

    with open(output_file, "w") as f:
        json.dump(res_pred, f, indent=4)





            

if __name__ == '__main__':
    main()
