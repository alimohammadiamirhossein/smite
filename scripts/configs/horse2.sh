# mod_config.sh

# Declare associative arrays for different mods
declare -A checkpoint_paths
declare -A video_paths
declare -A gt_base_paths
declare -A train_dirs
declare -A val_dirs
declare -A file_names
declare -A prompts
declare -A regularization_weight
declare -A track_weight

############################################ Training Setups ############################################
# Texual prompt for the mod
prompts["horse2"]="background nose head neck body1 body2 legs1 legs2 tail"

train_dirs["horse2"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/h2/train_10"
val_dirs["horse2"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/h2/val"

min_crop_ratio["horse2"]=0.8


############################################ Inference Setups ############################################
# Assign values for different mods (e.g., cars, horses)
checkpoint_paths["horse2"]="/localhome/aaa324/Project/FLATTEN/SMiTe/SMITE/logs/log_horse2_2024-10-14_11-07-27/ckpt_best.pt"

video_base_paths["horse2"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/h/vi/"

file_names["horse2"]="horse2 horse3 horse20 horse9 horse43 horse45 horse46 horse48 horse2 horse64 horse2 Horse_3_front_view Horse_2_front_view"
file_names["horse2"]="horse9"

gt_base_paths["horse2"]=""


regularization_weight["horse2"]=0
track_weight["horse2"]=0

