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
prompts["horse3"]="background head tail body1 body2 leg1 leg2 leg3 leg4 leg5"

train_dirs["horse3"]="data/h3/train_10"
val_dirs["horse3"]="data/h3/val"

min_crop_ratio["horse3"]=0.8


############################################ Inference Setups ############################################
# Assign values for different mods (e.g., cars, horses)
checkpoint_paths["horse3"]="logs/log_horse3_2024-10-13_12-37-03/ckpt_best.pt"

video_base_paths["horse3"]="data/h/vi/"

file_names["horse3"]="horse2 horse3 horse20 horse9 horse43 horse45 horse46 horse48 horse2 horse64 horse2 Horse_3_front_view Horse_2_front_view"
file_names["horse3"]="horse9"

gt_base_paths["horse3"]=""


regularization_weight["horse3"]=0
track_weight["horse3"]=0

