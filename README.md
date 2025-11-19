<h1 align="center">DEMO: Dense Motion Captioning</h1>

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

<p align="center">
<b>3DV 2026</b>
</p>
sorry for the delay but i'm really working on updating everything🥹


## TODO

<details>
<summary>TODO List: </summary>

- [x] update [website page](https://xusy2333.com/demo/)
- [x] README
- [x] release [Dataset](https://huggingface.co/datasets/Xusy2333/DEMO) <<< need to be updated but at least the `.npy` is released 
- [x] train scripts for DEMO
- [ ] train scripts for UniMotion
- [ ] evaluation script (but eval seems to be isolated, i don't remember)
- [ ] pretrained weights (on huggingface?)
- [ ] dataset generation scripts
- [ ] maybe? some ablation exp and setting? if someone has interests
</details>

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

3. you need to modify the `motion` path in `stage1.json` and `stage2.json`. you need to re-generated the json.

```
# need to replace the downloaded dataset path. 
python data/generate_json.py 
```

now you can train the model!

### Maybe the Process of CompMo Generation

tbc. this part of code and script depend on [STMC](https://github.com/nv-tlabs/stmc). so maybe you can also generate you own dense kind of data. 

### Pretrained Weights

tbd, will release in some google drive. (or huggingface?)

## Train Scripts (On CompMo)


### Stage1: motion-language alignment on humanml3d

just need to modify your `HumanmL3D/new_joints` path
```
deepspeed train.py --deepspeed scripts/zero2.json --freeze_backbone True --conv_type plain --tune_mm_mlp_adapter True --data_path datasets/stage1.json --motion_folder YOUR_JOINTS_ROOT_OF_HUMANML3D --data_root YOUR_JOINTS_ROOT_OF_HUMANML3D --motion_dim 1056 --exp_name stage1 --output_dir logs/stage1 --log_base logs --vision_tower mlp
```

### Stage2: dense captioning instruction turning on compmo

need to modify your compmo path
```
deepspeed train.py --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero2.json --conv_type llama_3 --pretrain_mm logs/stage1/mm_projector.bin \
    --group_by_modality_length True --exp_name stage2 --output_dir logs/stage2 \
    --data_path datasets/stage2.json --motion_folder YOUR_COMPMO_PATH --data_root YOUR_COMPMO_PATH \
    --motion_dim 1056 --num_train_epochs 3 --gradient_accumulation_steps 2 --per_device_train_batch_size 4 \
    --evaluation_strategy no --save_strategy steps --save_steps 5000 --save_total_limit 1 \
    --learning_rate 2e-5 --warmup_ratio 0.1 --lr_scheduler_type cosine --logging_steps 1 \
    --model_max_length 3072 --gradient_checkpointing True --max_grad_norm=1.0 \
    --dataloader_num_workers 4 --lazy_preprocess True --bf16 True --vision_tower mlp \
    --log_base logs --model_name_or_path logs/stage1
```

### Inference

```
python inference.py --model_path logs/stage2 --data datasets/test.json --output datasets/results/test.json --motion_dim 1056
```


### On H3D+BABEL (UniMotion setting)

## Evaluate Scripts

## Citation

```
@article{xu2025densemotioncaptioning,
      title={Dense Motion Captioning}, 
      author={Shiyao Xu and Benedetta Liberatori and Gül Varol and Paolo Rota},
      journal={arXiv preprint arXiv:2511.05369},
      url={https://arxiv.org/abs/2511.05369}, 
      year={2025}
    }
```

