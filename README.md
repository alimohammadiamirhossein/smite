
<div align="center">

<h1>SMITE: Segment Me In TimE</h1>

<div>
    <a href='https://alimohammadiamirhossein.github.io/' target='_blank'>Amir Alimohammadi</a><sup>1</sup>&emsp;
    <a href='https://sauradip.github.io/' target='_blank'>Sauradip Nag</a><sup>1</sup>&emsp;
  <a href='https://asgsaeid.github.io/' target='_blank'>Saeid Asgari Taghanaki</a><sup>1, 2</sup>&emsp; 
  <a href='https://taiya.github.io/' target='_blank'>Andrea Tagliasacchi</a><sup>1, 3, 4</sup>&emsp; <br>
  <a href='https://www.medicalimageanalysis.com/' target='_blank'>Ghassan Hamarneh</a><sup>1</sup>&emsp;
    <a href='https://www.sfu.ca/~amahdavi' target='_blank'>Ali Mahdavi Amiri</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1<b>Simon Fraser University</b>
    <sup>2</sup>Autodesk Research
    <sup>3</sup>University of Toronto
    <sup>4</sup>Google DeepMind
        &emsp; <br>
</div>
<div>
    Accepted in <b>ICLR 2025</b>
</div>

<h3 align="center">
  <a href="https://arxiv.org/abs/2410.18538" target='_blank'>Paper</a> |
  <a href="https://segment-me-in-time.github.io" target='_blank'>Project Page</a> 
</h3>
</div>
SMITE is an advanced open-source framework for temporally consistent video segmentation, designed to predict and segment objects across video frames using one or few reference images. With its ability to track and generalize unseen video sequences based on flexible granularity, it ensures precision and efficiency in segmentation, even when faced with occlusions, varying poses, and lighting conditions.

<div align="center">
<table>
<tr>
    <td><img src="https://github.com/alimohammadiamirhossein/smite/blob/main/assets/teaser.png" width="100%"/></td>
</tr>
</table>
</div>

## Note: We will release the full code soon. Stay tuned for updates!

## Why SMITE?

- SMITE minimizes dataset needs by leveraging pre-trained diffusion models and a few reference images for segmentation.
- It ensures consistent segmentation across video frames with its advanced tracking and temporal voting mechanism.
- It offers flexible segmentation at various granularities, making it ideal for tasks requiring different levels of detail.

We invite you to explore the full potential of SMITE and look forward to your feedback and contributions. If you find SMITE valuable, kindly star the repository on GitHub!

# Overview

The main parts of the framework are as follows:

```
SMITE
├── run.py                            -- script to train the models, runs factory.trainer.py                      
├── models                    
│   ├── unet.py                       -- inflated unet definition
|   ├── attention.py                  -- FullyFrameAttention to attend to all frames
│   ├── controlnet3d.py               -- ControlNet3D model definition (it is not part of the main model but we support it)
|   ├── ...
├── src
|   ├── pipeline_smite.py             -- main pipeline containing all the important functions
|   ├── train.py                     
|   ├── inference.py                     
|   ├── slicing.py                    -- slices frames and latents for efficient attention processing across video sequences
|   ├── tracking.py                   -- tracker initialization, applies tracking to each frame, and uses feature voting
|   ├── frequency_filter.py           -- DCT filter for low-pass regularization
|   ├── metric.py                     
|   ├── latent_optimization.py        -- spatio-temporal guidance          
|   ├── ...   
├── scripts
|   ├── train.sh                      --script for model training
|   ├── inference.sh                  --script for model inference on videos
|   ├── test_on_images.sh             --script for testing the model on image datasets
|   ├── ...   
├── utils
|   ├── setup.py                     
|   ├── args.py                       -- define, parse, and update command-line arguments
|   ├── transfer_weights.py           -- transfer the 2D Unet weights to inflated Unet
|   ├── ...   
```
# Getting Started  
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, evaluate your pretrained models, and produce visualizations.  

### Dependencies
- Python 3.8+
- PyTorch == 2.1.1 **(Please make sure your pytorch version is atleast 1.8)**
- NVIDIA GPU 3090TX
- Hugging-Face Diffusers
- xformers == 0.0.23

### Environment Setup
You can create and activate a Conda environment like below:
```shell script
conda create -n <envname> python=3.8
conda activate <envname>  
pip install --upgrade pip
```

### Requirements  
Furthermore, you just have to install all the packages you need:  
```shell script  
pip install -r requirements.txt  
```  

# Usage

### Model Training 

To train SMITE from scratch run the following command. The training configurations can be adjusted from  ``` scripts/configs/car.sh ``` file.
```shell script
bash scripts/train.sh [domain of the objects (e.g., car, horse)]
```
for more training configs please visit [here](https://github.com/alimohammadiamirhossein/Division_Examples/blob/main/assets/ARGS_README.md#Train).

### Model Inference 
We will provide the pretrained models containing the checkpoints for the following classes with granularity (1=coarse, 3=fine): 
| Class | Granularity | Link | 
|:---:|:---:|:---:|
  | Cars | - | [ckpt](link) |
| Horses | 1,2,3 |  [ckpt1](link),[ckpt2](link),[ckpt3](link) |
| Faces | 1,2,3 |  [ckpt1](link),[ckpt2](link),[ckpt3](link) |

After downloading the checkpoints or training the model yourself, set the checkpoints path in ``` --ckpt_path=/path/to/ckpt_best.pt ``` file.
The model inference can be then performed using the following command 

```shell script
bash scripts/inference.sh \
[domain of the objects (e.g., car, horse)] \
--ckpt_path=/path/to/ckpt_best.pt
--video_path=/path/to/video.mp4
```
for more information about inference please visit [here](https://github.com/alimohammadiamirhossein/Division_Examples/blob/main/assets/ARGS_README.md#Inference).


## Optional : Test on Image datasets

To test SMITE on image dataset like PASCAL-Parts run the following command.
```shell script
bash scripts/test_on_images.sh \
[domain of the objects (e.g., car, horse)] \
--ckpt_path=/path/to/ckpt_best.pt \
--test_dir=/path/to/the/dataset
```

### TO-DO Checklist

- [x] Add training script
- [x] Add inference script
- [ ] Add dataset
- [ ] Support for XMEM++ dataset
- [ ] Enable multi-gpu training


## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@misc{alimohammadi2024smitesegmenttime,
      title={SMITE: Segment Me In TimE}, 
      author={Amirhossein Alimohammadi and Sauradip Nag and Saeid Asgari Taghanaki and Andrea Tagliasacchi and Ghassan Hamarneh and Ali Mahdavi Amiri},
      year={2024},
      eprint={2410.18538},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.18538}, 
}
```
