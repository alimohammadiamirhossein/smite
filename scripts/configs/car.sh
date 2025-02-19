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
prompts["car"]="background body light plate wheel window"

train_dirs["car"]="/localhome/aaa324/Project/FLATTEN/Clean_Version/Division_Examples/data/car/train_10"
val_dirs["car"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/car/val"
test_dirs["car"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/car/test"

min_crop_ratio["car"]=0.6


############################################ Inference Setups ############################################
# Assign values for different mods (e.g., cars, horses)
checkpoint_paths["car"]="/localhome/aaa324/Project/FLATTEN/SMiTe/SMITE/logs/log_car_2024-11-14_21-50-57/ckpt_best.pt"

video_base_paths["car"]="/localhome/aaa324/Project/FLATTEN/Clean_Version/Division_Examples/data/car/vi/"

file_names["car"]="car42 car7 car2 car17 car3 car11 car18 car22 car23 car25"
# file_names["car"]="car17 car3 car11 car18 car22 car23 car25 car2"
# file_names["car"]="car42 car7"   

gt_base_paths["car"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/ground_truth/car"


regularization_weight["car"]=0
track_weight["car"]=0

