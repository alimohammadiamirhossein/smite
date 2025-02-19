import torch
import argparse


def add_basic_args(parser):
    parser.add_argument("--prompt", type=str, default="", help="Textual prompt for video editing")
    parser.add_argument("--neg_prompt", type=str, default="", help="Negative prompt for guidance")
    parser.add_argument("--guidance_scale", default=10.0, type=float, help="Guidance scale")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--dataset_name", type=str, default="pascal", help="Name of dataset")
    parser.add_argument("--train_dir", type=str, default="/localhome/aaa324/Project/FLATTEN/Clean_Version/VideoSLIME/data/car/train_10", help="Directory of training data")
    parser.add_argument("--val_dir", type=str, default="/localhome/aaa324/Project/FLATTEN/Clean_Version/VideoSLIME/data/car/test", help="Directory of validation data")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Directory of output")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base", help="Model ID of Stable Diffusion")
    parser.add_argument("--old_qk", type=int, default=0, help="Whether to use old queries and keys for flow-guided attention")
    parser.add_argument("--sample_steps", type=int, default=50, help="Steps for feature injection")
    parser.add_argument("--inject_step", type=int, default=40, help="Steps for feature injection")
    parser.add_argument("--use_controlnet", action="store_true", help="Whether to use ControlNet")
    parser.add_argument("--validating_on_images", action="store_true", help="Whether to validate")

def add_inference_arguments(parser):
    parser.add_argument("--gt_path", type=str, default=None, help="Path to ground truth")
    parser.add_argument("--modulation", action="store_true", help="Whether to use modulation")
    parser.add_argument("--track_weight", type=float, default=0.0, help="Whether to optimize based on tracking")
    parser.add_argument("--dct_threshold", type=float, default=0.4, help="Threshold for DCT")
    parser.add_argument("--regularization_weight", type=float, default=0.0, help="Whether to optimize based on reference segmentation")
    parser.add_argument("--not_inflated_unet", action="store_true", help="Whether to use simple UNet")
    parser.add_argument("--hierarchical_segmentation", action="store_true", help="Whether to use hierarchical segmentation")


def add_training_arguments(parser):
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--num_of_epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--coef_loss_sd", type=float, default=0.005, help="Coefficient of loss SD")
    parser.add_argument("--coef_loss_fourier", type=float, default=0.001, help="Coefficient of loss Fourier")
    parser.add_argument("--min_crop_ratio", type=float, default=0.8, help="Minimum crop ratio")
    parser.add_argument("--training_tokens", action="store_true", help="Whether to train tokens")
    parser.add_argument("--training_mode", type=str, default="text_embeds", help="Training mode")
    parser.add_argument("--training_time_interval", nargs="+", type=int, default=[5, 100], help="Time interval for training")
    parser.add_argument("--no_flip_dataset", action="store_true", help="Whether to flip dataset")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint")

def add_video_arguments(parser):
    parser.add_argument("--video_path", type=str, default=None, help="Path to a source video")
    parser.add_argument("--video_length", type=int, default=15, help="Length of output video")
    parser.add_argument("--frame_rate", type=int, default=None, help="The frame rate of loading input video. Default rate is computed according to video length.")
    parser.add_argument("--fps", type=int, default=15, help="FPS of the output video")
    parser.add_argument("--starting_frame", type=int, default=0, help="Starting frame of loading input video")
    parser.add_argument("--height", type=int, default=512, help="Height of synthesized video, and should be a multiple of 32")
    parser.add_argument("--width", type=int, default=512, help="Width of synthesized video, and should be a multiple of 32")

def add_miscellaneous_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model")
    parser.add_argument("--seed", type=int, default=7, help="Random seed of generator")

def add_attention_layers_args(parser):
    parser.add_argument(
        "--attention_layers_to_use",
        nargs="+",
        type=str,
        default=[
            "up_blocks[1].attentions[0].transformer_blocks[0].attn2",
            "up_blocks[1].attentions[1].transformer_blocks[0].attn2",
            "up_blocks[1].attentions[2].transformer_blocks[0].attn2",
            "up_blocks[2].attentions[0].transformer_blocks[0].attn2",
            "up_blocks[2].attentions[1].transformer_blocks[0].attn2",
            "up_blocks[3].attentions[0].transformer_blocks[0].attn1",
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1",
            "up_blocks[3].attentions[2].transformer_blocks[0].attn1"
        ],
    )

def add_layers_to_save_args(parser):
    parser.add_argument(
        "--layers_to_save",
        nargs="+",
        type=str,
        default=[
            "up_blocks[1].resnets[0].out_layers_features",
            "up_blocks[1].resnets[1].out_layers_features",
            "up_blocks[2].resnets[0].out_layers_features",
            "up_blocks[1].attentions[0].transformer_blocks[0].attn2.q",
            "up_blocks[1].attentions[0].transformer_blocks[0].attn2.k",
            "up_blocks[1].attentions[1].transformer_blocks[0].attn2.q",
            "up_blocks[1].attentions[1].transformer_blocks[0].attn2.k",
            "up_blocks[1].attentions[2].transformer_blocks[0].attn2.q",
            "up_blocks[1].attentions[2].transformer_blocks[0].attn2.k",
            "up_blocks[2].attentions[0].transformer_blocks[0].attn2.q",
            "up_blocks[2].attentions[0].transformer_blocks[0].attn2.k",
            "up_blocks[2].attentions[1].transformer_blocks[0].attn2.q",
            "up_blocks[2].attentions[1].transformer_blocks[0].attn2.k",
            "up_blocks[3].attentions[0].transformer_blocks[0].attn1.q",
            "up_blocks[3].attentions[0].transformer_blocks[0].attn1.k",
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1.q",
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1.k",
            "up_blocks[3].attentions[2].transformer_blocks[0].attn1.q",
            "up_blocks[3].attentions[2].transformer_blocks[0].attn1.k"
        ],
    )


def get_args():
    parser = argparse.ArgumentParser()

    add_basic_args(parser)
    add_training_arguments(parser)
    add_video_arguments(parser)
    add_miscellaneous_arguments(parser)
    add_attention_layers_args(parser)
    add_layers_to_save_args(parser)
    add_inference_arguments(parser)

    args = parser.parse_args()
    return args

def load_and_update_args(args, ckpt_path=None):
    if ckpt_path is None:
        return args
        
    hparams = torch.load(ckpt_path)
    args.learning_rate = hparams["learning_rate"]
    args.use_controlnet = hparams["use_controlnet"]
    args.coef_loss_fourier = hparams["coef_loss_fourier"]
    args.coef_loss_sd = hparams["coef_loss_sd"]
    args.training_mode = hparams["training_mode"]
    args.dataset_name = hparams["dataset_name"]

    return args

