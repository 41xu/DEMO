deepspeed train.py --deepspeed scripts/zero2.json --freeze_backbone True --conv_type plain --tune_mm_mlp_adapter True --data_path datasets/stage1.json --motion_folder /home/sxu/HumanML3D/HumanML3D/new_joints --data_root /home/sxu/HumanML3D/HumanML3D/new_joints/ --motion_dim 1056 --exp_name stage1 --output_dir logs/stage1 --log_base logs --vision_tower mlp

deepspeed train.py --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero2.json --conv_type llama_3 \
    --pretrain_mm logs/stage1/mm_projector.bin \
    --group_by_modality_length True --exp_name stage2 \
    --output_dir logs/stage2 \
    --data_path datasets/stage2.json \
    --motion_folder /home/sxu/stmc/final/ \
    --data_root /home/sxu/stmc/final/ \
    --motion_dim 1056 --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 4 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --max_grad_norm=1.0 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --bf16 True \
    --vision_tower mlp \
    --log_base logs \
    --model_name_or_path logs/stage1


# python inference.py --model_path logs/stage2 --data datasets/test.json --output datasets/results/test.json --motion_dim 1056
