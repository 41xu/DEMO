<h1 align="center">[3DV 2026] DEMO: Dense Motion Captioning</h1>

<p align="center">
  <a href='https://arxiv.org/abs/2511.05369'>
    <img src='https://img.shields.io/badge/Arxiv-Pdf-A42C25?style=flat&logo=arXiv&logoColor=white'></a>
  <a href='https://xusy2333.com/demo/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=white'></a>
  <a href='https://huggingface.co/datasets/Xusy2333/DEMO'> 
    <img src='https://img.shields.io/badge/Dataset-DEMO-blue?logo=huggingface&logoColor=white'></a>
</p>

<p align="center">
<a href="https//xusy2333.com/">Shiyao Xu</a>, <a href="https://benedettaliberatori.github.io/">Benedetta Liberatori</a>, <a href="https://gulvarol.github.io/">Gül Varol</a>, <a href="https://paolorota.github.io/">Paolo Rota</a>
<!-- [Shiyao Xu](https://xusy2333.com), [Benedetta Liberatori](https://benedettaliberatori.github.io/), [Gül Varol](https://gulvarol.github.io/), [Paolo Rota](https://paolorota.github.io/) -->
</p>

sorry for the delay but i'm really working for updating everything🥹

I'm updating everything this week 😭 the code is really messy 🚬

## TODO

- [x] update [website page](https://xusy2333.com/demo/)
- [ ] README
- [x] release [Dataset](https://huggingface.co/datasets/Xusy2333/DEMO) <<< need to be updated but at least the `.npy` is released 
- [ ] train scripts for DEMO
- [ ] train scripts for UniMotion
- [ ] evaluation script (but eval seems to be isolated, i don't remember)
- [ ] pretrained weights (on huggingface?)
- [ ] dataset generation scripts
- [ ] maybe? some ablation exp and setting? if someone has interests

## Environment Setup

for conda: 
```bash
conda create python=3.9 -n demo  # it's finally 3.9 since my env is broken. 3.10+ should also work
conda activate demo
# torch==2.6.0+cu124 or sth else
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
pip install torch torchvision sentencepiece peft einops fastapi gradio numpy openai opencv_python pillow ray requests shortuuid tqdm uvicorn scipy bitsandbytes deepspeed tensorboard
# i prefer transformers==4.44.0
pip install transformers==4.44.0 
pip install flash-attn --no-build-isolation
```

## Data Prepare

### HumanML3D

please follow [HumanML3D](https://github.com/EricGuo5513/HumanML3D), actually download [AMASS](https://amass.is.tue.mpg.de/) dataset, and follow H3D steps. note that we only need `new_joints` info, so maybe no `motion_representation` script in H3D needed.

### CompMo 

download CompMo dataset from [huggingface](https://huggingface.co/datasets/Xusy2333/DEMO). 

vis (mp4) and the data (npy files) can also be found in this google drive (<<<tbd, since the mp4s are really large, so...).

### After Download

1. put humanml3d and compmo anywhere you want.

2. make sure you also download `stageX.json` and put it in `datasets/`

3. you need to modify the `motion` path in `stage1.json` and `stage2.json`, or you need to re-generated the json.

now you can train the model!

### Maybe the Process of CompMo Generation

tbc. this part of code and script depend on [STMC](https://github.com/nv-tlabs/stmc). so maybe you can also generate you own dense kind of data. 

### Pretrained Weights

tbd, will release in some google drive. (or huggingface?)

