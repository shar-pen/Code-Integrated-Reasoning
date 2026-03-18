# for SFT, you have to move files of global_step_?/huggingface to fsdp global_step_? directories first

python3 -m scripts.legacy_model_merger merge \
	--backend fsdp \
	--local_dir /home/pengxia3/dc/ckpts/GRPO_math_CIR/Qwen3-4B-Thinking-2507_SFT_0316_v2_step@40_v1/global_step_50/actor \
	--target_dir /home/pengxia3/dc/ckpts/GRPO_math_CIR/Qwen3-4B-Thinking-2507_SFT_0316_v2_step@40_v1/global_step_50/actor_merged