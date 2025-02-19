#!/bin/bash

mod_name="$1"

# Source configuration and argument parsing scripts
source ./scripts/configs/$mod_name.sh
source ./scripts/parser/check_args_inference.sh
source ./scripts/parser/parse_args_inference.sh "$@"

LOG_DIR="./old_logs/${mod_name}_final"
file="${file_names[$mod_name]}"  # Only one video file

# Set the parameter ranges
# regularization_weights=(8)
# track_weights=(2 1)
regularization_weights=(0)
track_weights=(0)
video_length=24

for regularization_weight in "${regularization_weights[@]}"; do
  for track_weight in "${track_weights[@]}"; do
    output_name="${file}_reg${regularization_weight}_track${track_weight}"

    cmd="python run.py \
        --guidance_scale \"$guidance_scale\" \
        --video_path \"${video_base_paths[$mod_name]}/${file}.mp4\" \
        --ckpt_path \"$ckpt_path\" \
        --starting_frame \"$starting_frame\" \
        --frame_rate \"$frame_rate\" \
        --video_length \"$video_length\" \
        --output_path \"vis/video_outputs/$output_name\" \
        --prompt \"${prompts[$mod_name]}\" \
        --dct_threshold \"$dct_threshold\" \
        --regularization_weight \"$regularization_weight\" \
        --track_weight \"$track_weight\""

    if [ "$hierarchical_segmentation" == "true" ]; then
      cmd="$cmd --hierarchical_segmentation"
    fi

    if [[ -n "${gt_base_path}" ]]; then
      gt_path="${gt_base_path}/${file}"
      cmd="$cmd --gt_path \"$gt_path\""
    fi

    # Execute the command
    eval $cmd
  done
done