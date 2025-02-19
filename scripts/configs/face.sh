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
prompts["face"]="background skin eye mouth nose brow ear neck cloth hair"

# train_dirs["face"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/celeba/train_10"
train_dirs["face"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/celeba/train_1"
val_dirs["face"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/celeba/val"
test_dirs["face"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/celeba/test"

min_crop_ratio["face"]=0.6


############################################ Inference Setups ############################################
# Assign values for different mods (e.g., cars, horses)
checkpoint_paths["face"]="/localhome/aaa324/Project/FLATTEN/SMiTe/SMITE/logs/log_face_2024-11-14_11-51-48/ckpt_best.pt"

video_base_paths["face"]="/localhome/aaa324/Project/FLATTEN/Clean_Version/Division_Examples/data/celeba/vi/"

file_names["face"]="face5 face7 face9 face11 face12 face13 face14 face17 face21 face23 face24 face26 face27 face28"
# file_names["face"]="face5"

gt_base_paths["face"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/ground_truth/face"

regularization_weight["face"]=0
track_weight["face"]=0

