# scripts execution

## infer

CUDA_VISIBLE_DEVICES='7' bash exp_scripts/run_infer.sh 

## sft

NGPUS_PER_NODE=4 PROJ_NAME="cir_sft" EXP_NAME="qwen_2.5_1.5b" CUDA_VISIBLE_DEVICES='0,1,2,3' bash exp_scripts/run_sft.sh



