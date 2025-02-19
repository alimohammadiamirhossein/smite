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
prompts["non_semantic"]="background skin1 skin2 skin3 eye brow lip ear"

train_dirs["non_semantic"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/complex/train_5"
val_dirs["non_semantic"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/complex/val"
test_dirs["non_semantic"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/complex/val"

min_crop_ratio["non_semantic"]=0.6


############################################ Inference Setups ############################################
# Assign values for different mods (e.g., cars, horses)
# checkpoint_paths["non_semantic"]="/localhome/aaa324/Project/FLATTEN/SMiTe/SMITE/logs/log_non_semantic_2024-11-19_18-23-11/ckpt_best.pt"
checkpoint_paths["non_semantic"]="/localhome/aaa324/Project/FLATTEN/SMiTe/SMITE/logs/log1_5_10/log_non_semantic_train_5/ckpt_best.pt"

video_base_paths["non_semantic"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/data/half_face/vi"

# file_names["face"]="face5 face7 face9 face11 face12 face13 face14 face17 face21 face23 face24 face26 face27 face28"
file_names["non_semantic"]="half_face9 half_face11 half_face12 half_face13 half_face14 half_face15 half_face16 half_face28 half_face35"

gt_base_paths["non_semantic"]="/localhome/aaa324/Project/FLATTEN/DiViSe_Custom/ground_truth/non_semantic"

regularization_weight["non_semantic"]=0
track_weight["non_semantic"]=0

