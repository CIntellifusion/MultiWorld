export OMP_NUM_THREADS=32
python -m torch.distributed.run --nproc_per_node=8 \
    robots/parallel_inference.py \
    --config-path robots/configs/inference.yaml \
    --model-path checkpoints/multiworld_320p_robots.safetensors \
    --output-dir outputs/eval_robotics