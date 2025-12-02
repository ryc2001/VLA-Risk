"""
run_single_task_custom_instruction.py

Simplified script for testing a single task with custom instructions.
Evaluates one task with custom instruction input and calculates success rate.

Usage:
    python experiments/robot/libero/run_single_task_custom_instruction.py \
        --model_family openvla \
        --pretrained_checkpoint openvla-7b-finetuned-libero-goal \
        --task_suite_name libero_goal \
        --task_id 0 \
        --custom_instruction "put the bowl on the stove" \
        --num_trials 10 \
        --center_crop True \
        --local_log_dir "./experiments/logs"
"""

import os

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import torch
from libero.libero import benchmark
from PIL import Image, ImageDraw, ImageFont

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
sys.path.append(os.path.abspath("."))        # 保证能找到当前目录
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class SingleTaskConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # Load with 8-bit quantization
    load_in_4bit: bool = False                       # Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # Task-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"             # Task suite
    task_id: int = 0                                 # Specific task ID to test
    num_trials: int = 20                             # Number of trials for this task
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize

    #################################################################################################################
    # Custom instruction parameters
    #################################################################################################################
    custom_instruction: str = ""                     # Custom instruction to use instead of task description
    use_custom_instruction: bool = True              # Whether to use custom instruction

    #################################################################################################################
    # Image saving parameters
    #################################################################################################################
    save_final_images: bool = True                   # Whether to save final step images
    images_dir: str = "./experiments/logs/"    # Directory to save final images

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for logs
    seed: int = 7                                    # Random seed

    # fmt: on


def save_final_image(img: np.ndarray, task_description: str, save_path: str) -> None:
    """Save the final image with task description as filename info."""
    # Convert numpy array to PIL Image
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save image
    pil_img.save(save_path)
    print(f"[INFO] Saved final image: {save_path}")


@draccus.wrap()
def eval_single_task_custom_instruction(cfg: SingleTaskConfig) -> None:
    """Evaluate a single task with custom instruction input."""
    
    # Validate inputs
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    
    if cfg.use_custom_instruction:
        assert cfg.custom_instruction, "cfg.custom_instruction must be provided when use_custom_instruction=True!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    print("Loading model...")
    model = get_model(cfg)
    print("=== MODEL INFO ===")
    print("Model type:", type(model))
    print("Pretrained checkpoint:", cfg.pretrained_checkpoint)
    print("Model device:", next(model.parameters()).device)
    print("=== NORM_STATS KEYS ===")
    print(list(model.norm_stats.keys()))
    print("Using unnorm_key:", cfg.unnorm_key)
    print("norm_stats[cfg.unnorm_key]:")
    print(model.norm_stats.get(cfg.unnorm_key, None))


    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize logging
    run_id = f"SINGLE-TASK-{cfg.task_suite_name}-{cfg.task_id}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    run_id += "--custom-instruction"
    
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    
    # Validate task_id
    assert 0 <= cfg.task_id < task_suite.n_tasks, f"Task ID {cfg.task_id} out of range [0, {task_suite.n_tasks})"
    
    # Get specific task
    task = task_suite.get_task(cfg.task_id)
    original_task_description = task.language
    
    # Use custom instruction if specified
    if cfg.use_custom_instruction:
        task_description = cfg.custom_instruction
        print(f"Using custom instruction: {task_description}")
        print(f"Original task description: {original_task_description}")
        log_file.write(f"Using custom instruction: {task_description}\n")
        log_file.write(f"Original task description: {original_task_description}\n")
    else:
        task_description = original_task_description
        print(f"Using original task description: {task_description}")
        log_file.write(f"Using original task description: {task_description}\n")
    # === Create folder for saving step images (same structure as custom_image script) ===
    task_images_dir = f"{cfg.local_log_dir}/images_with_text/task_{cfg.task_id:02d}_{task_description.replace(' ', '_').replace('.', '_')[:30]}"
    os.makedirs(task_images_dir, exist_ok=True)

    # Get initial states for this task
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    print(f"Available initial states: {len(initial_states)}")
    log_file.write(f"Available initial states: {len(initial_states)}\n")

    # Initialize LIBERO environment
    env, _ = get_libero_env(task, cfg.model_family, resolution=256)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Set max steps based on task suite
    if cfg.task_suite_name == "libero_spatial":
        max_steps = 220
    elif cfg.task_suite_name == "libero_object":
        max_steps = 280
    elif cfg.task_suite_name == "libero_goal":
        max_steps = 300
    elif cfg.task_suite_name == "libero_10":
        max_steps = 520
    elif cfg.task_suite_name == "libero_90":
        max_steps = 400
    else:
        max_steps = 300  # default

    # Start evaluation
    total_episodes, total_successes = 0, 0
    
    for episode_idx in tqdm.tqdm(range(cfg.num_trials), desc="Evaluating"):
        print(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")

        # Reset environment
        env.reset()

        # Set initial states
        obs = env.set_init_state(initial_states[episode_idx])

        # Setup
        t = 0
        replay_images = []
        done = False

        print(f"Starting episode {episode_idx+1}...")
        log_file.write(f"Starting episode {episode_idx+1}...\n")
        
        while t < max_steps + cfg.num_steps_wait:
            try:
                # Wait for objects to stabilize
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Get preprocessed image (always use real-time simulation image)
                img = get_libero_image(obs, resize_size)
                                # === Save an image every 5 steps (same logic as custom_image script) ===
                # === Save an image at the beginning and then every 5 steps ===
                # 开始的图片：t == cfg.num_steps_wait
                # 之后每 5 步： (t > cfg.num_steps_wait) 且 (t - cfg.num_steps_wait) % 5 == 0
                if t == cfg.num_steps_wait or (t > cfg.num_steps_wait and (t - cfg.num_steps_wait) % 5 == 0):
                    episode_dir = f"{task_images_dir}/episode_{episode_idx+1:02d}"
                    os.makedirs(episode_dir, exist_ok=True)
                    save_path = f"{episode_dir}/step_{t:03d}.jpg"
                    print(f"[INFO] Saving image at step {t}: {save_path}")

                    save_img = img
                    if save_img.dtype != np.uint8:
                        save_img = (save_img * 255).clip(0, 255).astype("uint8")
                    pil_img = Image.fromarray(save_img)
                    pil_img.save(save_path)

                # Save preprocessed image for replay video
                replay_images.append(img)

                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                # Query model to get action
                action = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,  # Use custom instruction here
                    processor=processor,
                )

                # Normalize gripper action [0,1] -> [-1,+1]
                action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] Flip gripper action sign
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                
                if done:
                    total_successes += 1
                    break
                    
                t += 1

            except Exception as e:
                print(f"Caught exception: {e}")
                log_file.write(f"Caught exception: {e}\n")
                break

        total_episodes += 1

        # Save final step image if enabled
        if cfg.save_final_images and t > cfg.num_steps_wait:
            # Get the final image
            final_img = get_libero_image(obs, resize_size)
            
            # Create filename with task info
            task_name = task_description.replace(" ", "_").replace(",", "").replace(".", "")[:50]  # Limit length
            filename = f"episode_{episode_idx+1:02d}_final_{task_name}.jpg"
            save_path = os.path.join(cfg.images_dir, filename)
            
            # Save the final image
            save_final_image(final_img, task_description, save_path)

        # Save a replay video of the episode
        save_rollout_video(
            replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
        )

        # Log current results
        print(f"Success: {done}")
        print(f"# episodes completed so far: {total_episodes}")
        print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        log_file.write(f"Success: {done}\n")
        log_file.write(f"# episodes completed so far: {total_episodes}\n")
        log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
        log_file.flush()

    # Final results
    final_success_rate = total_successes / total_episodes * 100 if total_episodes > 0 else 0
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Custom instruction: {task_description}")
    print(f"Original task: {original_task_description}")
    print(f"Total trials: {total_episodes}")
    print(f"Successes: {total_successes}")
    print(f"Success rate: {final_success_rate:.1f}%")
    
    log_file.write(f"\n=== FINAL RESULTS ===\n")
    log_file.write(f"Custom instruction: {task_description}\n")
    log_file.write(f"Original task: {original_task_description}\n")
    log_file.write(f"Total trials: {total_episodes}\n")
    log_file.write(f"Successes: {total_successes}\n")
    log_file.write(f"Success rate: {final_success_rate:.1f}%\n")
    
    # Close environment and log file
    env.close()
    log_file.close()
    
    print(f"\nResults saved to: {local_log_filepath}")


if __name__ == "__main__":
    eval_single_task_custom_instruction()

