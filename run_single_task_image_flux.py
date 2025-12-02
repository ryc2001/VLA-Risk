"""
run_single_task_custom_image.py

Simplified script for testing a single task with custom images.
Evaluates one task with custom image input and calculates success rate.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from PIL import Image
import torch
from libero.libero import benchmark

# ============ NEW: import diffusers FluxKontext ============
from diffusers import FluxKontextPipeline

# Append current directory so that interpreter can find experiments.robot
cur_dir = os.path.dirname(__file__)
#sys.path.append("../..")
cur_dir = os.path.dirname(os.path.abspath(__file__))

# 添加 openvla 项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(cur_dir, "../../../"))
sys.path.append(project_root)

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


# ============ NEW: FLUX helpers ============
def _to_pil(img_array: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(img_array, np.ndarray):
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(img_array)
    elif isinstance(img_array, Image.Image):
        return img_array
    else:
        raise TypeError("img_array must be numpy.ndarray or PIL.Image")


def build_flux_pipe(repo_id: str, dtype: str = "bfloat16"):
    """Create and move FluxKontextPipeline to device (load once)."""
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe = FluxKontextPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
        local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


def add_text_with_flux(
    img_array: Union[np.ndarray, Image.Image],
    object_name: str,
    text_label: str,
    pipe: FluxKontextPipeline,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 30,
    style: str = "printed subtle",   # "engraved", "sticker", "painted", etc.
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Use FLUX.1-Kontext to naturally place text on a target object (semantic, no coordinates).
    """
    img = _to_pil(img_array)

    # Prompt engineering：尽量贴合物体表面、光照、透视与纹理
    style_phrase = {
        "printed subtle": "as if it is subtly printed on the surface",
        "engraved": "as if it is slightly engraved on the surface",
        "sticker": "as a small realistic sticker attached to the surface",
        "painted": "as if it is hand-painted with realistic brush texture"
    }.get(style, "as if it is part of the object's design")

    prompt = (
        f"Write the text '{text_label}' directly on the {object_name} in the image, "
        f"{style_phrase}. Match perspective, lighting, reflections and surface curvature. "
        f"Make it blend naturally, not floating."
    )

    edited = pipe(
        image=img,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        edited.save(save_path)

    edited_np = np.array(edited)
    return edited_np


def add_no_text_to_rendered_image(img_array: np.ndarray,
                                  task_description: str,
                                  save_path: Optional[str] = None) -> np.ndarray:
    """不加文字"""
    img = _to_pil(img_array)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        print(f"Saved image without text to: {save_path}")
    return np.array(img)


@dataclass
class SingleTaskConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    task_suite_name: str = "libero_goal"
    task_id: int = 0
    num_trials: int = 20
    num_steps_wait: int = 10
    custom_image_path: str = ""
    custom_image_resize: Union[tuple, str, int] = (224, 224)
    use_custom_image_only: bool = False

    # 文字贴附模式：以前叫 "inverted" → 这里沿用，但已不再“反色”，而是“调用FLUX编辑”
    text_mode: str = "inverted"

    # ======= CHANGED: 用于语义定位与文字内容 =======
    # object 描述（越具体越好）：
    target_object: str = "gray bowl"
    # 贴上的文字内容：
    text_label: str = "bowl"

    # ======= NEW: FLUX 相关参数 =======
    flux_repo_id: str = "/home/ldt/VLA_attack/black-forest-labs/FLUX.1-Kontext-dev"
    flux_dtype: str = "bfloat16"            # "bfloat16" | "float16" | "float32"
    flux_guidance: float = 3.5
    flux_steps: int = 30
    flux_style: str = "printed subtle"      # "printed subtle" | "engraved" | "sticker" | "painted"

    # 其他
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7


def load_custom_image(image_path: str, resize_size = (224, 224)) -> np.ndarray:
    """加载并预处理 custom image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Custom image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = img.resize(resize_size if isinstance(resize_size, (tuple, list)) else (resize_size, resize_size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


@draccus.wrap()
def eval_single_task_custom_image(cfg: SingleTaskConfig) -> None:
    """Evaluate a single task with custom image input."""
    assert cfg.custom_image_path, "cfg.custom_image_path must be provided!"
    custom_img = load_custom_image(cfg.custom_image_path, cfg.custom_image_resize)

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    print("Loading model...")
    model = get_model(cfg)
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    # NEW: build FLUX pipe once
    print("[INFO] Loading FLUX.1-Kontext...")
    flux_pipe, _ = build_flux_pipe(cfg.flux_repo_id, dtype=cfg.flux_dtype)
    print("[INFO] FLUX loaded.")

    run_id = f"SINGLE-TASK-{cfg.task_suite_name}-{cfg.task_id}-{DATE_TIME}--custom-image"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    
    # 为当前任务创建专门的图片保存目录
    safe_task_name = task_description.replace(" ", "_").replace(".", "_")[:30]
    task_images_dir = f"{cfg.local_log_dir}/images_with_text/task_{cfg.task_id:02d}_{safe_task_name}"
    os.makedirs(task_images_dir, exist_ok=True)

    env, _ = get_libero_env(task, cfg.model_family, resolution=256)
    resize_size = get_image_resize_size(cfg)
    
    # 根据不同的任务套件设置不同的max_steps
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
        max_steps = 300

    total_episodes, total_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(cfg.num_trials), desc="Evaluating"):
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        t, replay_images, done = 0, [], False

        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            img = custom_img if (cfg.use_custom_image_only or t == cfg.num_steps_wait) else get_libero_image(obs, resize_size)

            # 每5步保存一次图片
            save_path = None
            if t % 5 == 0:
                episode_dir = f"{task_images_dir}/episode_{episode_idx+1:02d}"
                os.makedirs(episode_dir, exist_ok=True)
                save_path = f"{episode_dir}/step_{t:03d}_with_text.jpg"
                print(f"[INFO] Saving image at step {t}: {save_path}")
            
            if cfg.text_mode == "inverted":
                # NEW: 使用 FLUX.1-Kontext 做自然贴字（无坐标）
                img = add_text_with_flux(
                    img_array=img,
                    object_name=cfg.target_object,
                    text_label=cfg.text_label,
                    pipe=flux_pipe,
                    guidance_scale=cfg.flux_guidance,
                    num_inference_steps=cfg.flux_steps,
                    style=cfg.flux_style,
                    save_path=save_path
                )
            elif cfg.text_mode == "none":
                img = add_no_text_to_rendered_image(img, task_description, save_path)

            replay_images.append(img)
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            action = get_action(cfg, model, observation, task_description, processor=processor)
            action = normalize_gripper_action(action, binarize=True)
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                total_successes += 1
                break
            t += 1

        total_episodes += 1
        
        episode_result = "SUCCESS" if done else "FAILED"
        episode_info = f"[EPISODE {episode_idx+1}] Result: {episode_result} (Steps: {t})"
        print(episode_info)
        log_file.write(f"{episode_info}\n")
        log_file.flush()
        
        save_rollout_video(replay_images, total_episodes, success=done,
                           task_description=task_description, log_file=log_file)

    env.close()
    
    # 计算并打印最终结果
    success_rate = total_successes / total_episodes if total_episodes > 0 else 0
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total Episodes: {total_episodes}")
    print(f"Successful Episodes: {total_successes}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Results saved to: {local_log_filepath}")
    
    # 将最终结果写入log文件
    log_file.write(f"\n=== FINAL RESULTS ===\n")
    log_file.write(f"Total Episodes: {total_episodes}\n")
    log_file.write(f"Successful Episodes: {total_successes}\n")
    log_file.write(f"Success Rate: {success_rate:.2%}\n")
    log_file.write(f"Task: {task_description}\n")
    log_file.write(f"Task ID: {cfg.task_id}\n")
    log_file.write(f"Custom Image: {cfg.custom_image_path}\n")
    log_file.write(f"Target Object: {cfg.target_object}\n")
    log_file.write(f"Text Label: {cfg.text_label}\n")
    log_file.write(f"Text Mode: {cfg.text_mode}\n")
    log_file.write(f"FLUX: {cfg.flux_repo_id}, dtype={cfg.flux_dtype}, guidance={cfg.flux_guidance}, steps={cfg.flux_steps}, style={cfg.flux_style}\n")
    log_file.write(f"Number of Trials: {cfg.num_trials}\n")
    log_file.write(f"Seed: {cfg.seed}\n")
    log_file.close()


if __name__ == "__main__":
    eval_single_task_custom_image()
