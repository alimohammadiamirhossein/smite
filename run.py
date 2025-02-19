import os
import torch
from utils import (
                    get_args,
                    setup_output_directory,
                    adjust_dimensions,
                    load_pipeline,
                    setup_training_pipeline,
                    setup_validation_pipeline,
                    setup_inference_pipeline,
                    set_generator,
                    load_and_update_args,
                    load_video,
                    get_crop_coords_if_needed,
                   )

def pipeline_training(args, pipe, generator, dtype=torch.float32):
    setup_training_pipeline(pipe, input_prompt=args.prompt, min_crop_ratio=args.min_crop_ratio, dataset_name=args.dataset_name,
                                train_dir=args.train_dir, val_dir = args.val_dir, flip_dataset=not args.no_flip_dataset,
                                batch_size=args.batch_size, training_mode=args.training_mode, 
                                learning_rate=args.learning_rate, dtype=dtype)
        
    pipe.fit(args.prompt, num_of_epochs=args.num_of_epochs,
                video_length=args.video_length, generator=generator, guidance_scale=args.guidance_scale,
                negative_prompt=args.neg_prompt, width=args.width, height=args.height,
                token_ids=list(range(pipe.num_parts)), attention_output_size=(64, 64), apply_softmax=False,
                trajs=None, output_dir="tmp/", inject_step=args.inject_step, old_qk=args.old_qk, inter_frame=True, 
                time_interval=args.training_time_interval, training_mode=args.training_mode,
                coef_loss_sd=args.coef_loss_sd, coef_loss_fourier=args.coef_loss_fourier)

def pipeline_validation(args, pipe, generator, crop_coords=None):
    if args.validating_on_images:
        setup_validation_pipeline(pipe, input_prompt=args.prompt, min_crop_ratio=args.min_crop_ratio, dataset_name=args.dataset_name,
                                train_dir=args.train_dir, val_dir = args.val_dir, batch_size=args.batch_size, training_mode=args.training_mode,
                                ckpt_path=args.ckpt_path)

        pipe.validate(args.prompt, video_length=args.video_length, generator=generator, guidance_scale=args.guidance_scale,
            negative_prompt=args.neg_prompt, width=args.width, height=args.height,
            token_ids=list(range(pipe.num_parts)), attention_output_size=(64, 64), apply_softmax=False,
            trajs=None, output_path="tmp/", inject_step=args.inject_step, old_qk=args.old_qk, inter_frame=True, 
            time_interval=args.training_time_interval, training_mode=args.training_mode, crop_coords=crop_coords)
    else:
        pipeline_inference(args, pipe, generator, crop_coords=crop_coords)

def pipeline_inference(args, pipe, generator, crop_coords=None):
    setup_inference_pipeline(pipe, args.prompt, ckpt_path=args.ckpt_path)
    real_frames, real_depths, gt_dict = load_video(args.video_path, args.video_length, args.width, args.height,
                                            frame_rate=args.frame_rate, use_controlnet=args.use_controlnet,
                                            gt_path=args.gt_path ,starting_frame=args.starting_frame, 
                                            hierarchical_segmentation=args.hierarchical_segmentation)

    inference_args = {'video_length': args.video_length if gt_dict.get('new_video_length', None) is None else gt_dict['new_video_length'], 
                    'frames': real_frames, 'depths': real_depths, 'height': args.height, 
                    'width': args.width, 'num_inference_steps': args.sample_steps, 'generator': generator, 
                    'guidance_scale': args.guidance_scale,  'negative_prompt': args.neg_prompt, 'token_ids': list(range(pipe.num_parts)),
                    'attention_output_size': (64, 64), 'apply_softmax': False, 'output_path': args.output_path, 
                    'inject_step': args.inject_step, 'old_qk': args.old_qk, 'layers_to_save': args.layers_to_save, 
                    'training_mode': args.training_mode, 'track_weight': args.track_weight, 
                    'gt': gt_dict.get('gt', None),  
                    'dct_threshold': args.dct_threshold, 'inflated_unet': not args.not_inflated_unet,
                    'regularization_weight': args.regularization_weight, 'inter_frame': True}

    if args.modulation:
        output = pipe.inference_with_modulation(args.prompt, crop_coords=crop_coords, **inference_args)
    elif args.hierarchical_segmentation:
        output = pipe.inference_with_hierarchical_segmentation(args.prompt, crop_coords=crop_coords, **inference_args)
    else:
        output = pipe(args.prompt, crop_coords=crop_coords, **inference_args)

    if gt_dict.get('gt', None) is not None:
        pipe.calculate_metrics(gt_dict['gt'], output['segmentation'].argmax(dim=1), 
                               gt_inds=gt_dict['gt_inds'], frame_inds=gt_dict['frame_inds'],
                               output_path=args.output_path)

def main():
    args = get_args()
    device = args.device
    adjust_dimensions(args)
    
    crop_coords = get_crop_coords_if_needed(args, patch_size=400)
    
    # generator = set_generator(args.seed, device)
    generator = set_generator(777, device)

    if args.training_tokens:
        pipe = load_pipeline(args.model_id, device, args.attention_layers_to_use, args.use_controlnet, dtype=torch.float32)
        # pipeline_training(args, pipe, generator, dtype=torch.float16 if args.use_controlnet else torch.float32)
        pipeline_training(args, pipe, generator, dtype=torch.float16)
    else:
        args = load_and_update_args(args, ckpt_path=args.ckpt_path)
        # crop_coords = get_crop_coords_if_needed(args, patch_size=350)
        crop_coords = get_crop_coords_if_needed(args, patch_size=450)
        pipe = load_pipeline(args.model_id, device, args.attention_layers_to_use, args.use_controlnet, dtype=torch.float16)
        pipeline_validation(args, pipe, generator, crop_coords=crop_coords)

        
if __name__ == "__main__":
    main()
