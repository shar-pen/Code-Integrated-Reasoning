# for SFT, you have to move files of global_step_?/huggingface to fsdp global_step_? directories first

python3 -m scripts.legacy_model_merger merge \
	--backend fsdp \
	--local_dir /home/pengxia3/dc/ckpts/GRPO_math_CIR/Qwen2.5-3B-Instruct_SFT_step@10_3rd/global_step_50/actor \
	--target_dir /home/pengxia3/dc/ckpts/GRPO_math_CIR/Qwen2.5-3B-Instruct_SFT_step@10_3rd/global_step_50_merged