# Define variables for columns
export SOURCE_COLUMN="ghost_image"
export TARGET_COLUMN="target"
export CAPTION_COLUMN="Prompt"
export MODEL_NAME="black-forest-labs/FLUX.1-Kontext-dev"
export TRAIN_DATASET_NAME="raresense/SAKS_Jewelry"
export VAL_DATASET_NAME="raresense/SAKS_Jewelry_test"
export OUTPUT_DIR="SAKS_Lora_Training_Kontext_dev"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$TRAIN_DATASET_NAME \
  --validation_dataset_name=$VAL_DATASET_NAME \
  --source_column=$SOURCE_COLUMN \
  --target_column=$TARGET_COLUMN \
  --caption_column=$CAPTION_COLUMN \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --rank=128 \
  --gradient_accumulation_steps=8 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10050 \
  --num_train_epochs=5 \
  --seed="42" \
  --height=512 \
  --width=512 \
  --max_sequence_length=512  \
  --checkpointing_steps=2000  \
  --validation_check \
  --validation_steps=1000 \
  --report_to="wandb" \
  # --resume_from_checkpoint="latest"  \