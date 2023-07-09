export CUDA_VISIBLE_DEVICES=0,1,2
TRAIN_FILE=../mix.data.txt
EVAL_FILE=../mix.data.txt


python run_train.py \
    --output_dir=$OUTPUT_DIR \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --seed 42 \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --extraction 'softmax' \
    --align_layer 9 \
    --train_wc \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --weight_decay 0 \
    --softmax_threshold 0.001 \
    --softmax_omit_threshold 1e-20 \
    --temperature 1 \
    --save_steps 5000 \
    --max_steps 20000 \
    --logging_steps 100 \
    --warmup_steps 200 \
    --do_train \
    --eval_data_file=$EVAL_FILE \
    --overwrite_output_dir