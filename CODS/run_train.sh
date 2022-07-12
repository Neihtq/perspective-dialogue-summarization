gpu=$1

# CUDA_VISIBLE_DEVICES=$gpu python hf_train_bart.py --do_train --do_segment --gen_keyphrase_summary --output_dir=save/bart-large-xsum-samsum-segment-genkpsum

# CUDA_VISIBLE_DEVICES=$gpu python hf_train_bart.py --do_train --gen_keyphrase_summary --output_dir=save/bart-large-xsum-samsum-genkpsum  --test_target_max_len=400 --target_max_len=400 --train_batch_size 2 --eval_batch_size 2 --validation_timing 5000
# CUDA_VISIBLE_DEVICES=0,1 python hf_train_bart.py --do_train --gen_keyphrase_summary --output_dir=save/bart-large-xsum-dialoguesum-genkpsum  --test_target_max_len=400 --target_max_len=400 --train_batch_size 2 --eval_batch_size 2 --validation_timing 5000

#TRAIN_PATH=data/processed/new_dialogsum_shuffled/train.json
#VAL_PATH=data/processed/new_dialogsum_shuffled/eval.json
#TEST_PATH=data/processed/new_dialogsum_shuffled/test.json
#OUTPUT_DIR=models/penalty_1
TRAIN_PATH=data/processed/new_dialogsum_clean_data/train.json
VAL_PATH=data/processed/new_dialogsum_clean_data/eval.json
TEST_PATH=data/processed/new_dialogsum_clean_data/test.json
OUTPUT_DIR=models/penalty_salesforce_5
#MODEL_NAME=facebook/bart-large-xsum
MODEL_NAME=Salesforce/bart-large-xsum-samsum
#python main.py --do_train --model_name=Salesforce/bart-large-xsum-samsum \
#                --output_dir=models/dialogsum_new_1 --train_file_path=data/new_dialogsum/new_clean_data/train.json \
#                --dev_file_path=data/new_dialogsum/new_clean_data/eval.json --test_file_path=data/new_dialogsum/new_clean_data/test.json \
#                --num_train_epochs=50 --target_max_len=400 --patience=20 \
#                --gen_keyphrase_summary --train_batch_size=1 --eval_batch_size=1 --validation_timing=1
#CUDA_VISIBLE_DEVICES=0,1
python main.py --do_train --model_name=$MODEL_NAME \
                --output_dir=$OUTPUT_DIR --train_file_path=$TRAIN_PATH \
                --dev_file_path=$VAL_PATH --test_file_path=$TEST_PATH \
                --num_train_epochs=500 --target_max_len=400 --patience=100 \
                --gen_keyphrase_summary --train_batch_size=16 --eval_batch_size=16 --validation_timing=1 --penalty-term=0.8

#CUDA_VISIBLE_DEVICES=1 python main.py --model_name=$MODEL_NAME \
#                --output_dir=$OUTPUT_DIR --train_file_path=$TRAIN_PATH \
#                --dev_file_path=$VAL_PATH --test_file_path=$TEST_PATH \
#                --num_train_epochs=50 --target_max_len=400 --patience=30 \
#                --gen_keyphrase_summary --train_batch_size=1 --eval_batch_size=1 --validation_timing=1 --penalty-term=1
