#!/bin/bash

mod_name="$1"

source ./scripts/configs/$mod_name.sh
source ./scripts/parser/check_args_inference.sh
source ./scripts/parser/parse_args_inference.sh "$@"


output_filename="./vis/test_on_image_result.txt"

# Build the command
cmd="python run.py \
    --train_dir \"${train_dirs[$mod_name]}\" \
    --val_dir \"${test_dir}\" \
    --video_path \"${video_base_paths[$mod_name]}/car7.mp4\" \
    --prompt \"${prompts[$mod_name]}\" \
    --ckpt_path \"${checkpoint_paths[$mod_name]}\" \
    --width \"512\" \
    --height \"512\" \
    --validating_on_images"

# Run the command
eval $cmd > $output_filename
    