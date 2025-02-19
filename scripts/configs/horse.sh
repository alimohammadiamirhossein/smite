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
prompts["horse"]="background head neck+torso leg tail"

# train_dirs["horse"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/h/train_10"
train_dirs["horse"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/h/train_1"
val_dirs["horse"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/h/val"
test_dirs["horse"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/h/test"

min_crop_ratio["horse"]=0.8


############################################ Inference Setups ############################################
# Assign values for different mods (e.g., cars, horses)
checkpoint_paths["horse"]="logs/log_horse_2024-11-12_19-47-09/ckpt_best.pt"

video_base_paths["horse"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/h/vi/"

file_names["horse"]="horse3 horse20 horse9 horse45 horse46 horse48 Horse_3_front_view Horse_2_front_view horse2"

gt_base_paths["horse"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/ground_truth/horse"


regularization_weight["horse"]=0
track_weight["horse"]=0

