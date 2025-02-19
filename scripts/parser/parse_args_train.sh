#!/bin/bash

mod_name="$1"
shift

source ./scripts/configs/$mod_name.sh
source ./scripts/common/common_train_params.sh

min_crop_ratio="${min_crop_ratio[$mod_name]}"

use_controlnet="false"
no_flip_dataset="false"

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --guidance_scale=*) guidance_scale="${1#*=}" ;;
      --num_of_epochs=*) num_of_epochs="${1#*=}" ;;
      --seed=*) seed="${1#*=}" ;;
      --width=*) width="${1#*=}" ;;
      --height=*) height="${1#*=}" ;;
      --training_time_interval=*) training_time_interval="${1#*=}" ;;
      --coef_loss_sd=*) coef_loss_sd="${1#*=}" ;;
      --coef_loss_fourier=*) coef_loss_fourier="${1#*=}" ;;
      --learning_rate=*) learning_rate="${1#*=}" ;;
      --training_mode=*) training_mode="${1#*=}" ;;
      --use_controlnet=*) use_controlnet="${1#*=}" ;;
      --no_flip_dataset=*) no_flip_dataset="${1#*=}" ;;
      *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
  done
}

# Parse command-line arguments and possibly override the values
parse_args "$@"


