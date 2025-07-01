# Define variables for columns
export SOURCE_COLUMN="source"
export TARGET_COLUMN="target"
export CAPTION_COLUMN="caption"
export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export TRAIN_DATASET_NAME="raresense/Viton"
# export TEST_DATASET_NAME="raresense/Viton_validation"
export OUTPUT_DIR="saved_weights"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$TRAIN_DATASET_NAME \
  --source_column=$SOURCE_COLUMN \
  --target_column=$TARGET_COLUMN \
  --caption_column=$CAPTION_COLUMN \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --seed="42" \
  --height=512 \
  --width=512 \
  --max_sequence_length=512  \
  --checkpointing_steps=2500  \
  --report_to="wandb" \
  # --resume_from_checkpoint="latest"  \