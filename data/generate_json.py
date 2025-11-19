import os
import json
from tqdm import tqdm
from random import sample
import random
import torch
import numpy as np

timeformat = "Identify the description of each motion in the given sequence in the format 'mm:ss:ms - text'. \n<motion>"



# instructions = [
#     "Generate a textual description corresponding to the given sequence of human motion.\n<motion>",
#     # "Explain the motion illustrated in <motion> using language.",
#     "Explain the motion using language.\n<motion>",
#     "Describe the motion.\n<motion>",
#     # "Describe the action being represented by <motion> using text.",
#     "Give me a brief summary of the movement depicted in <motion>.",
#     # "Explain the motion illustrated in <motion> using text.",
#     "Please explain the movement being represented by <motion> using text.",
#     "Describe the actions by using text.\n<motion>",
#     "Build a narrative description that matches the stated series of human motion cues.\n<motion>",
#     "What is the sequence of movements the person is performing in the motion?\n<motion>",
#     "Form a written description that correlates with the seies of human motion provided.\n<motion>",
#     "Translate the given human motion into a corresponding textual description.\n<motion>",
#     "Describe the motion in natural language.\n<motion>",
# ]

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

instruct_new = ("<motion>\n\nGiven a complex human motion sequence of duration {} which includes several actions, describe these actions in the motion with natural language according to the movement of human. \n" \
    "The description of each action should be in the format 'text # start # end'. \n" \
    "Here is an example: moves in a curve to the right side # 0.0 # 5.1,  doing a left foot squat # 5.1 # 7.0\n")


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
                # "value": random.choice(instructions)
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
    prepare_stage1(outfile="./datasets/stage1.json")
    # motion_folder = path of compomo
    prepare_comp(motion_folder="/home/sxu/stmc/final", texts_file="datasets/overall_timelines.txt", out_file="datasets/stage2.json")
    prepare_comp_val(motion_folder="/home/sxu/stmc/final", texts_file="datasets/overall_timelines.txt", out_file="datasets/test.json", split="test")
