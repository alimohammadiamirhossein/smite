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
prompts["davis"]="background object"

# train_dirs["davis"]="Davis2016/train_data"
train_dirs["davis"]="Davis2016/single_instance/soapbox"
val_dirs["davis"]="Davis2016/single_instance/soapbox"
test_dirs["davis"]="Davis2016/test_data"

min_crop_ratio["davis"]=0.8


############################################ Inference Setups ############################################
# Assign values for different mods (e.g., cars, horses)
checkpoint_paths["davis"]="logs/log_davis_soapbox/ckpt_best.pt"

video_base_paths["davis"]="Davis2016/DAVIS_test"

# file_names["davis"]="blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox"
file_names["davis"]="soapbox"

gt_base_paths["davis"]="Davis2016/DAVIS_test/ground_truth"


regularization_weight["davis"]=0
track_weight["davis"]=0

