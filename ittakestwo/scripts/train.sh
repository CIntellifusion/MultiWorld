CONFIG_PATH="${1}"
EXP_NAME="${2}"
DEBUG_FLAG="${3:-""}"
OUTPUT_DIR="./outputs/$EXP_NAME"
mkdir -p $OUTPUT_DIR


echo "[ItTakesTwoSimulator] Config: $CONFIG_PATH"
echo "[ItTakesTwoSimulator] Experiment Name: $EXP_NAME"
echo "[ItTakesTwoSimulator] Output Directory: $OUTPUT_DIR"
echo "[ItTakesTwoSimulator] Debug Mode: $DEBUG_FLAG"

accelerate launch ittakestwo/train.py \
  --config_path $CONFIG_PATH \
  --output_path $OUTPUT_DIR \
  --trainable_models "dit" \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --extra_inputs "input_image" ${DEBUG_FLAG}