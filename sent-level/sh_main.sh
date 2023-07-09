export CUDA_VISIBLE_DEVICES=0

python main_d.py \
    --model_type $MODEL_PATH \
    --seed 42 \
    --train_data_file=$TRAIN_FILE \
    --permuted_file=$PERMUTED_FILE \
    --output_dir=$OUTPUT_DIR \
    --beta 1 \
    --alpha 30 \
    --ratio 0.5 \
    --type 'permuted' \
    --permute_num 7 \
    --batch_size 5 \
    --gradient_accumulation_steps 2 \
    --logging_steps 100 \
    --teacher_batch_size 500 \
    --num_epoch 1 \
    --max_steps 500 \
    --weight_decay 0 \
    --learning_rate 1e-6 \
    --warmup_steps 1000 \
    # --teacher_model_type $TEACHER_MODEL_TYPE \