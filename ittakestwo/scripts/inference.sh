export OMP_NUM_THREADS=32
python -m torch.distributed.run --nproc_per_node=8 \
    ittakestwo/parallel_inference.py \
    --inference-seed 0 \
    --num-inference-steps 50 \
    --config-path ittakestwo/configs/inference_480P_full.yaml \
    --model-path checkpoints/multiworld_480p_fulldata.safetensors \
    --output-dir outputs/eval_480P_full 

python -m torch.distributed.run --nproc_per_node=8 \
    ittakestwo/parallel_inference.py \
    --inference-seed 0 \
    --num-inference-steps 35 \
    --config-path ittakestwo/configs/inference_480P_toy.yaml \
    --model-path checkpoints/multiworld_480p_toydata.safetensors \
    --output-dir outputs/eval_480P_toy

