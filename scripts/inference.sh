#!/bin/bash

mod_name="$1"

source ./scripts/configs/$mod_name.sh
source ./scripts/parser/check_args_inference.sh
source ./scripts/parser/parse_args_inference.sh "$@"

LOG_DIR="./old_logs/${mod_name}_final"
filenames=(${file_names[$mod_name]})

for file in "${filenames[@]}"; do
  output_name="${file}"

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

  if [ "$not_inflated_unet" == "true" ]; then
    cmd="$cmd --not_inflated_unet"
  fi

  eval $cmd
done
