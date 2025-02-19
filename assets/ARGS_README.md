# Arguments
This is a description to provide details about arguments of SMITE. SMITE is an advanced open-source framework for temporally consistent video segmentation, designed to predict and segment objects across video frames using one or few reference images. It is implemented by Amir Alimohammadi.

# Train
```
SMITE
├── run.py                            -- script to train the models, runs factory.trainer.py                      
├── scripts                    
│   ├── train.sh                      -- train script
│   ├── common                        
│   │   ├── common_train_params.sh    -- common parameters for training
│   ├── parser                        
│   │   ├── check_args_train.sh       -- validates training-specific arguments (ensures no conflicts or missing parameters)
│   │   ├── parse_args_train.sh       -- parse command-line arguments and convert them to parameters for training
|   ├── ...
```

## Train Parameters:

These parameters are used to configure the behavior of the training script (`run.py`) for SMITE.

#### `--prompt`
- **Description**: A text input provided to guide the model during training. Although the model can learn even with random text, providing a meaningful prompt can improve the speed of convergence and enhance training efficiency.
- **Type**: `string`
- **Example**: `--prompt="background body light plate wheel window"`

#### `--learning_rate`
- **Description**: The learning rate for the optimizer.
- **Type**: `float`
- **Example**: `--learning_rate=1e-6`

#### `--num_of_epochs`
- **Description**: The number of epochs.
- **Type**: `int`
- **Default**: `300`
- **Example**: `--num_of_epochs=100`

#### `--coef_loss_sd`
- **Description**: The coefficient for the SD loss, as described in the SLiMe paper. It is primarily used to maintain the generalization ability of the diffusion model and prevent overfitting.
- **Type**: `float`
- **Default**: `0.005`
- **Example**: `--coef_loss_sd=0.01`

#### `--coef_loss_fourier`
- **Description**: Description: This loss coefficient improves training stability, as demonstrated in the "FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization" paper. It helps regulate frequency components during training, leading to more stable convergence.
- **Type**: `float`
- **Default**: `0.001`
- **Example**: `--coef_loss_fourier=0.001`

#### `--min_crop_ratio`
- **Description**: The minimum cropping ratio for training data augmentation. This value controls the smallest portion of the image that can be cropped during data augmentation. A higher value means larger portions of the original image are kept, while a lower value allows more aggressive cropping.
- **Type**: `float`
- **Default**: `0.8`
- **Example**: `--min_crop_ratio=0.8`

#### `--training_mode`
- **Description**: Defines the training mode, such as utilizing text embeddings, self-attention, or cross-attention mechanisms. You can choose from the following combinations:
  - `text_embeds`
  - `selfattention_q`
  - `selfattention_kv`
  - `crossattention_q`
  - `crossattention_kv`
  - `text_embeds+selfattention_q`
  - `text_embeds+selfattention_kv`
  - `text_embeds+crossattention_q`
  - `text_embeds+crossattention_kv`

These modes allow for various attention and embedding configurations, influencing how the model processes data. For further details on the effectiveness of these modes, refer to section 4.2 of our paper.

- **Type**: `string`
- **Default**: `"text_embeds+crossattn_kv"`
- **Example**: `--training_mode="text_embeds+selfattn_q"`

#### `--no_flip_dataset`
- **Description**: A flag to disable horizontal flipping of the dataset during data augmentation. When specified, no random flipping of images will occur. This is useful for cases where maintaining the original orientation of the data is important.
- **Type**: `boolean`
- **Default**: `false`
- **Example**: `--no_flip_dataset`

# Inference
```
SMITE
├── run.py                            -- script to train the models, runs factory.trainer.py                      
├── scripts                    
│   ├── inference.sh                      -- inference script
│   ├── common                        
│   │   ├── common_inference_params.sh    -- common parameters for inference
│   ├── parser                        
│   │   ├── check_args_inference.sh       -- validates inference-specific arguments (ensures no conflicts or missing parameters)
│   │   ├── parse_args_inference.sh       -- parse command-line arguments and convert them to parameters for inference
|   ├── ...
```

## Inference Parameters:

These parameters are used to configure the behavior of the inference script (`run.py`) for SMITE.

# Inference Parameters

#### `--video_path`
- **Description**: The path to the video file which shall be utilized during inference. This path must be precise, as it leads the model to its visual query.
- **Type**: `string`
- **Example**: `--video_path="path/to/video.mp4"`

#### `--video_length`
- **Description**: The length of the video, measured in frames. This value allows one to limit the scope of inference to a desired fragment of time.
- **Type**: `int`
- **Example**: `--video_length=500`

#### `--frame_rate`
- **Description**: The frame rate of the video in question, determining the temporal resolution at which the video shall be processed.
- **Type**: `float`
- **Example**: `--frame_rate=20`

#### `--starting_frame`
- **Description**: The frame from whence the model shall commence its analysis. This parameter allows for the model to begin inference not from the inception of the video, but from a chosen frame of interest.
- **Type**: `int`
- **Example**: `--starting_frame=100`

#### `--dct_threshold`
- **Description**: Threshold value for the Discrete Cosine Transform (DCT). It defines the sensitivity of the DCT operation, controlling the compression or preservation of details in the inference process.
- **Type**: `float`
- **Default**: `0.4`
- **Example**: `--dct_threshold=0.6`

#### `--regularization_weight`
- **Description**: A weight factor for Low-pass regularization, used to influence the optimization based on reference segmentation. It ensures the stability of inference by penalizing large deviations.
- **Type**: `float`
- **Default**: `0.0`
- **Example**: `--regularization_weight=8`

#### `--track_weight`
- **Description**: The weight assigned to the tracking mechanism, influencing the significance of object tracking during inference. It is advised not to set this value too high, as a lower weight allows the model to propagate tracking adjustments gradually and consistently across the entire video.
- **Type**: `float`
- **Example**: `--track_weight=0.1`

#### `--hierarchical_segmentation`
- **Description**: A flag to enable hierarchical segmentation, which processes video frames in a multi-level manner for enhanced precision and refinement of segment boundaries.
- **Type**: `boolean`
- **Example**: `--hierarchical_segmentation`

#### `--gt_path`
- **Description**: The path to ground truth data, which can be used for comparison or reference during inference. 
- **Type**: `string`
- **Example**: `--gt_path="path/to/ground_truth"`


