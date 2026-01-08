# for SFT, you have to move files of global_step_?/huggingface to fsdp global_step_? directories first

python3 -m scripts.legacy_model_merger merge \
	--backend fsdp \
	--local_dir ckpts/sft/qwen_2.5_1.5b/global_step_25 \
	--target_dir ckpts/sft/qwen_2.5_1.5b/global_step_25_merged