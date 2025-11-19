import os
import json
from tqdm import tqdm
from random import sample
import random
import torch
import numpy as np
import argparse

timeformat = "Identify the description of each motion in the given sequence in the format 'mm:ss:ms - text'. \n<motion>"


instructions = [
    "Generate a textual description corresponding to the given sequence of human motion.\n" + timeformat,
    "Explain the motion using language.\n" + timeformat,
    "Describe the motion.\n" + timeformat,
    "Give me a brief summary of the movement depicted in the motion.\n" + timeformat,
    "Describe the actions by using text.\n" + timeformat,
    "Build a narrative description that matches the stated series of human motion cues.\n" + timeformat,
    "What is the sequence of movements the person is performing in the motion?\n" + timeformat,
    "Form a written description that correlates with the seies of human motion provided.\n" + timeformat,
    "Translate the given human motion into a corresponding textual description.\n" + timeformat,
    "Describe the motion in natural language.\n" + timeformat,
]

instruct = ("<motion>\n\nGiven a complex human motion sequence of duration {} which includes several actions, describe these actions in the motion with natural language according to the movement of human. \n" \
    "The description of each action should be in the format 'mm:ss:ms - text'. \n" \
    "Here is an example: 00:00:00 - moves in a curve to the right side, 00:05:09 - doing a left foot squat \n")



"""
data should be:

{
    "id": "012181",
    "motion": "/home/sxu/HumanML3D/HumanML3D/new_joints/012181.npy",
    "conversations": [
        {
            "from": "human",
            "value": "Describe the motion with natural language.\n<motion>"
        },
        {
            "from": "gpt",
            "value": ""
        }
    ]
}
"""
def time_convert(timestep):
    # from seconds to MM:SS:MS
    t_float = float(timestep)
    min = int(t_float // 60)
    sec = int(t_float % 60)
    mill = int((t_float - int(t_float)) * 1000)
    # mill = int(timestep.split(".")[1])
    return f"{min:02d}:{sec:02d}:{mill:02d}"



def prepare_stage1(outfile="datasets/stage1.json", hml_file="/home/sxu/HumanML3D/HumanML3D/new_joints", split_file="/home/sxu/datasets/H3D/train_val.txt", text_file="/home/sxu/datasets/H3D/texts"):
    # use hml to to generate data used for stage1 training.
    dur = 0
    prompt = ("<motion>\n\nGiven a human motion sequence of duration {} describe the motion with natural language according to the movement of human. \n")
    # hml_file = "/home/sxu/datasets/H3D/new_joint_vecs"
    # split_file = "/home/sxu/datasets/H3D/train.txt"
    output = []
    with open(split_file, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        data_dict = {}
        line = line.strip()
        if not os.path.exists(os.path.join(hml_file, f"{line}.npy")): continue
        with open(os.path.join(text_file, f"{line}.txt"), 'r') as f:
            texts = f.readlines()
            texts = [x.split("#")[0].strip() for x in texts]
        data = np.load(os.path.join(hml_file, f"{line}.npy"))
        if data.ndim != 3:
            continue
        dur = data.shape[0]
        if dur < 20: continue
        data_dict["id"] = line
        data_dict["motion"] = os.path.join(hml_file, f"{line}.npy")
        data_dict["conversations"] = [
            {
                "from": "human",
                "value": prompt.format(time_convert(str(dur / 20)))
            },
            {
                "from": "gpt",
                "value": sample(texts, 1)[0]
            },
        ]
        output.append(data_dict)
    with open(outfile, "w", encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



def prepare_comp(motion_folder, texts_file="datasets/overall_timelines.txt", out_file="./datasets/stage2.json", split="train", fps=20):
    outputs = []

    # need to process timelines, get the texts mapping?
    # for timesteps, need to be changed from 14.64s, 64.68s this kind of format, to MM:SS:MS format
    with open(texts_file, 'r') as f:
        lines = f.readlines()
    timelines = []
    timeline = []
    for line in lines:
        if line == "\n":
            if not timeline: # empty line
                continue
            timelines.append(timeline) # timeline finish, append, then clear and got next timeline
            timeline = []
            continue
        line = line.strip()
        elements = [x.strip() for x in line.split("#")]
        text = elements[0]
        # we just use start time as timestepes
        # timestep = float(elements[1]) # start time
        timestep = elements[1]
        timestep = time_convert(timestep)
        timeline.append(timestep + " - " + text)
    if timeline:
        timelines.append(timeline)

    timelines = [", ".join(timeline) for timeline in timelines]

    print("overall timelines: ", len(timelines))
    """
    timelines: list [str]; each timeline in timelines looks like:
        00:00:00 - waves with his left hand up and down, 00:06:13 - walks forward slowly in a jagged line, \
        00:11:29 - walking towards the right in a curved path, 00:16:84 - jumps up and down twice, \
        00:21:22 - makes an s-motion with their hands & arms
    """

    with open(os.path.join(motion_folder, f"{split}.txt"), 'r') as f:
        mids = f.readlines()

    for mid in tqdm(mids):
        mid = mid.strip()
        data_dict = {}

        if not os.path.exists(os.path.join(motion_folder, f"{mid}.npy")): continue

        data = np.load(os.path.join(motion_folder, f"{mid}.npy"))
        data = torch.tensor(data, dtype=torch.float32)
        text = timelines[int(mid)]
        m_length = data.shape[0]
        data = data.reshape(m_length, -1)

        data_dict["id"] = mid
        data_dict["motion"] = os.path.join(motion_folder, f"{mid}.npy")
        data_dict["conversations"] = [
            {
                "from": "human",
                "value": instruct.format(time_convert(str(m_length / 20)))
            },
            {
                "from": "gpt",
                "value": text
            },
        ]
        outputs.append(data_dict)


    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)



def prepare_comp_val(motion_folder, texts_file="/home/sxu/stmc/test10k.txt", out_file="./datasets/comp/test10k.json", split="test", fps=20):
    output = []

    with open(texts_file, 'r') as f:
        lines = f.readlines()
    timelines = []
    timeline = []
    for line in lines:
        if line == "\n":
            if not timeline: # empty line
                continue
            timelines.append(timeline) # timeline finish, append, then clear and got next timeline
            timeline = []
            continue
        line = line.strip()
        elements = [x.strip() for x in line.split("#")]
        text = elements[0]
        # we just use start time as timestepes
        # timestep = float(elements[1]) # start time
        timestep = elements[1]
        timestep = time_convert(timestep)
        timeline.append(timestep + " - " + text)
    if timeline:
        timelines.append(timeline)

    timelines = [", ".join(timeline) for timeline in timelines]  

    with open(os.path.join(motion_folder, f"{split}.txt"), 'r') as f:
        mids = f.readlines()
    
    for mid in tqdm(mids):
        mid = mid.strip()
        data_dict = {}
        if not os.path.exists(os.path.join(motion_folder, f"{mid}.npy")): continue
        data = np.load(os.path.join(motion_folder, f"{mid}.npy"))
        data = torch.tensor(data, dtype=torch.float32)
        text = timelines[int(mid)]
        m_length = data.shape[0]
        data = data.reshape(m_length, -1)

        data_dict["id"] = mid
        data_dict["motion"] = os.path.join(motion_folder, f"{mid}.npy")
        data_dict["gt"] = text

        output.append(data_dict)
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_out", type=str, default="./datasets/stage1.json")
    parser.add_argument("--stage2_out", type=str, default="./datasets/stage2.json")
    parser.add_argument("--test_out", type=str, default="./datasets/test.json")
    parser.add_argument("--motion_folder", type=str, help="path of compmo / humanml3d", default="dataset/compmo")
    parser.add_argument("--text_file", type=str, help="path of overall_timelines", default="datasets/overall_timelines.txt")
    args = parser.parse_args()

    prepare_stage1(outfile=args.stage1_out) # outfile = "./datasets/stage1.json"
    # motion_folder = path of compomo
    prepare_comp(motion_folder=args.motion_folder, texts_file=args.text_file, out_file=args.stage2_out)
    prepare_comp_val(motion_folder=args.motion_folder, texts_file=args.text_file, out_file=args.test_out, split="test")