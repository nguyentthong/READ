python ./src/run.py \
        -model=multi_modal_bart \
        -log_name=multi_modal_bart \
        -gpus='1' \
        -learning_rate=1e-3 \
        -scheduler_lambda1=10 \
        -scheduler_lambda2=0.95 \
        -num_epochs=100 \
        -grad_accumulate=5 \
        -max_input_len=512 \
        -max_output_len=64 \
        -max_img_len=256 \
        -n_beams=5 \
        -n_shots=2000 \
        -random_seed=0 \
        -do_train=True \
        -do_test=False \
        -limit_val_batches=1 \
        -val_check_interval=1 \
        -img_lr_factor=5 \
        -use_forget_gate \
        -cross_attn_type=0 \
        -use_img_trans \
        # -fusion_in_decoding


find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf