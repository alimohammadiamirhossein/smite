#!/bin/bash

mod_name="$1"

source ./scripts/configs/$mod_name.sh
source ./scripts/parser/check_args_train.sh
source ./scripts/parser/parse_args_train.sh "$@"

cmd="python run.py \
      --dataset_name \"$mod_name\" \
      --prompt \"${prompts[$mod_name]}\" \
      --neg_prompt \"$neg_prompt\" \
      --guidance_scale \"$guidance_scale\" \
      --train_dir \"${train_dirs[$mod_name]}\" \
      --val_dir \"${val_dirs[$mod_name]}\" \
      --coef_loss_sd \"$coef_loss_sd\" \
      --coef_loss_fourier \"$coef_loss_fourier\" \
      --num_of_epochs \"$num_of_epochs\" \
      --learning_rate \"${learning_rate}\" \
      --training_mode \"${training_mode}\" \
      --output_path \"$output_path\" \
      --min_crop_ratio \"$min_crop_ratio\" \
      --width \"$width\" \
      --height \"$height\" \
      --training_time_interval ${training_time_interval[@]} \
      --seed \"$seed\" \
      --training_tokens"

if [ "$use_controlnet" == "true" ]; then
    cmd="$cmd --use_controlnet"
fi

if [ "$no_flip_dataset" == "true" ]; then
    cmd="$cmd --no_flip_dataset"
fi

eval $cmd