import numpy as np


def get_alpha_prev(pipeline, timestep):
    prev_timestep = timestep - pipeline.scheduler.config.num_train_timesteps // pipeline.scheduler.num_inference_steps
    alpha_prod_t_prev = pipeline.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else pipeline.scheduler.final_alpha_cumprod
    return alpha_prod_t_prev


def get_inverse_timesteps(pipeline, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)

    # safety for t_start overflow to prevent empty timsteps slice
    if t_start == 0:
        return pipeline.inverse_scheduler.timesteps, num_inference_steps
    timesteps = pipeline.inverse_scheduler.timesteps[:-t_start]

    return timesteps, num_inference_steps - t_start
