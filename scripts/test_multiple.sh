#!/bin/bash

output_filename="./vis/test_on_image_result.txt"

# Clear the output file to avoid appending to an old file
> $output_filename

# Iterate over each log folder in ./logs
for log_folder in ./logs/*; do
    if [[ -d "$log_folder" ]]; then
        # Extract the folder name and mode
        folder_name=$(basename "$log_folder")
        mode=$(echo "$folder_name" | cut -d'_' -f2) # Extract the second component (e.g., 'car', 'face')
        
        echo "mode = $mode"
        # Source the relevant config and common params based on mode
        source ./scripts/configs/$mode.sh
        test_dir="${test_dirs[$mod_name]}"

        # Define the checkpoint path
        ckpt_path="$log_folder/ckpt_best.pt"

        # Build the command
        cmd="python run.py \
            --train_dir \"${train_dirs[$mode]}\" \
            --val_dir \"${test_dir}\" \
            --video_path \"${video_base_paths[$mode]}/car7.mp4\" \
            --prompt \"${prompts[$mode]}\" \
            --ckpt_path \"$ckpt_path\" \
            --width \"512\" \
            --height \"512\" \
            --validating_on_images"

        # Run the command and append the output with the folder name as a label
        echo "name: $folder_name output:" >> $output_filename
        eval $cmd 2> /dev/null | grep -v 'Validation:' >> $output_filename
    fi
done
