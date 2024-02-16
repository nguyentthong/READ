python ./src/run.py \
        -model=multi_modal_bart \
        -val_save_file=./evaluation/temp_valid_file \
        -test_save_file=./evaluation/results/summaries.txt \
        -log_name=test \
        -gpus='1' \
        -learning_rate=3e-5 \
        -scheduler_lambda1=10 \
        -scheduler_lambda2=0.95 \
        -num_epochs=100 \
        -grad_accumulate=5 \
        -max_input_len=512 \
        -max_output_len=64 \
        -max_img_len=256 \
        -n_beams=5 \
        -random_seed=0 \
        -do_train=False \
        -do_test=True \
        -limit_val_batches=1 \
        -val_check_interval=1 \
        -img_lr_factor=5 \
        -checkpoint=./ckpts/finetuned_read_pvla.ckpt \
        -use_forget_gate \
        -cross_attn_type=0 \
        -use_img_trans \


find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf