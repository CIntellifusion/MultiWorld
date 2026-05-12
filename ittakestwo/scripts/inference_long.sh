python -m torch.distributed.run --nproc_per_node=8 \
      ittakestwo/parallel_inference.py \
      --inference-mode autoregressive \
      --num-chunks 3 \
      --config-path ittakestwo/configs/inference_480P_full_long.yaml \
      --model-path checkpoints/multiworld_480p_fulldata.safetensors \
      --output-dir outputs/autoregressive_longvideo

