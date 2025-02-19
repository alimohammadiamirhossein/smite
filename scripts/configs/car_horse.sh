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
prompts["car_horse"]="background body light plate wheel window head neck+torso leg tail"

train_dirs["car_horse"]="/localhome/aaa324/Project/Davis2016/car_and_horse/train_10"
val_dirs["car_horse"]="/localhome/aaa324/Project/Davis2016/car_and_horse/val"
test_dirs["car_horse"]="/localhome/aaa324/Project/Davis2016/car_and_horse/test"

min_crop_ratio["car_horse"]=0.8


############################################ Inference Setups ############################################
# Assign values for different mods (e.g., cars, horses)
checkpoint_paths["car_horse"]="logs/log_car_horse_2024-11-23_18-44-19/ckpt_best.pt"

video_base_paths["car_horse"]="/localhome/aaa324/Project/Davis2016/car_and_horse/vi"

file_names["car_horse"]="car_horse_0 car_horse_2 car_horse_3 car_horse_4"

# gt_base_paths["car_horse"]=


regularization_weight["car_horse"]=0
track_weight["car_horse"]=0

