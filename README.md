
<div align="center">

<h1>SMITE: Segment Me in Time</h1>

<div>
    <a href='https://alimohammadiamirhossein.github.io/' target='_blank'>Amir Alimohammadi</a><sup>1</sup>&emsp;
    <a href='https://sauradip.github.io/' target='_blank'>Sauradip Nag</a><sup>1</sup>&emsp;
  <a href='https://asgsaeid.github.io/' target='_blank'>Saeid Asgari Taghanaki</a><sup>2</sup>&emsp; 
  <a href='https://taiya.github.io/' target='_blank'>Andrea Tagliasacchi</a><sup>1</sup>&emsp; <br>
  <a href='https://www.medicalimageanalysis.com/' target='_blank'>Ghassan Hamarneh</a><sup>1</sup>&emsp;
    <a href='https://www.sfu.ca/~amahdavi' target='_blank'>Ali Mahdavi Amiri</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>School of Computing, Simon Fraser University, Canada&emsp;
    <sup>2</sup>Autodesk Research&emsp; <br>
</div>

<h3 align="center">
  <a href="" target='_blank'>Paper</a> |
  <a href="https://segment-me-in-time.github.io" target='_blank'>Project Page</a> 
</h3>
</div>
SMITE is an advanced open-source framework for temporally consistent video segmentation, designed to predict and segment objects across video frames using one or few reference images. With its ability to track and generalize unseen video sequences based on flexible granularity, it ensures precision and efficiency in segmentation, even when faced with occlusions, varying poses, and lighting conditions.

<div align="center">
<table>
<tr>
    <td><img src="https://github.com/alimohammadiamirhossein/Division_Examples/blob/main/assets/teaser.png" width="100%"/></td>
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

### TO-DO Checklist

- [ ] Add dataset
- [ ] Add training script
- [ ] Add inference script
- [ ] Support for XMEM++ dataset
- [ ] Enable multi-gpu training


## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@article{nag2022zero,
  title={SMITE:Segment Me in Time},
  author={Names},
  journal={arXiv e-prints},
  pages={arXiv--2207},
  year={2024}
}
```
