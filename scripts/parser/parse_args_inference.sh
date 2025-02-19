#!/bin/bash

mod_name="$1"
shift

source ./scripts/configs/$mod_name.sh
source ./scripts/common/common_inference_params.sh

regularization_weight="${regularization_weight[$mod_name]}"
track_weight="${track_weight[$mod_name]}"
gt_base_path="${gt_base_paths[$mod_name]}"

hierarchical_segmentation="true"
not_inflated_unet="false"
ckpt_path="${checkpoint_paths[$mod_name]}"
test_dir="${test_dirs[$mod_name]}"

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --guidance_scale=*) guidance_scale="${1#*=}" ;;
      --starting_frame=*) starting_frame="${1#*=}" ;;
      --frame_rate=*) frame_rate="${1#*=}" ;;
      --video_length=*) video_length="${1#*=}" ;;
      --dct_threshold=*) dct_threshold="${1#*=}" ;;
      --regularization_weight=*) regularization_weight="${1#*=}" ;;  
      --gt_base_path=*) gt_base_path="${1#*=}" ;;
      --track_weight=*) track_weight="${1#*=}" ;;
      --ckpt_path=*) ckpt_path="${1#*=}" ;;
      --test_dir=*) test_dir="${1#*=}" ;;
      --hierarchical_segmentation=*) hierarchical_segmentation="${1#*=}" ;;
      --not_inflated_unet=*) not_inflated_unet="${1#*=}" ;;
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


