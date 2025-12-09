# VLA-Risk: Benchmarking Vision-languageaction Models With Physical Robustness

![robot scene](case/new2.png)

## Table of Contents

- [Script Overview](#script-overview)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Usage](#usage)
  - [Instruction Evaluation](#instruction-evaluation)
  - [Image Evaluation](#image-evaluation-flux)
- [Parameter Reference](#parameter-reference)
- [Output Description](#output-description)

---
<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <img src="case/true.gif" width="30%">
  <img src="case/false.gif" width="30%">
</div>

## Script Overview

### 1. `run_single_task_custom_instruction.py`
Evaluates model performance with **instructions**. Allows replacing the original task description with carefully crafted non-executable instructions.


### 2. `run_single_task_image_flux.py`
Evaluates model performance with **images**. Uses FLUX.1-Kontext model to naturally add text labels on images.


---
## Requirements
Please follow `requirements-min.txt` to install.

---
### Instruction Evaluation
#### Basic Usage

```bash
python experiments/robot/libero/run_single_task_custom_instruction.py \
    --model_family openvla \
    --pretrained_checkpoint <checkpoint_path> \
    --task_suite_name <dataset_name> \
    --task_id <task_id> \
    --custom_instruction "<your_instruction>" \
    --num_trials <num_trials> \
    --local_log_dir <log_dir>
```

### Image Evaluation

#### Basic Usage

```bash
python experiments/robot/libero/run_single_task_image_flux.py \
    --model_family openvla \
    --pretrained_checkpoint <checkpoint_path> \
    --task_suite_name <dataset_name> \
    --task_id <task_id> \
    --custom_image_path <image_path> \
    --target_object "<object_description>" \
    --text_label "<text_to_add>" \
    --text_mode inverted \
    --num_trials <num_trials> \
    --local_log_dir <log_dir>
```
---

## Output Description

### Image Outputs
- **Intermediate images**: Saved in `{local_log_dir}/images_with_text/task_{task_id:02d}_{task_name}/episode_{episode_idx:02d}/step_{step:03d}.jpg`
- **Final images** (only for `run_single_task_custom_instruction.py`): Saved in `{images_dir}/episode_{episode_idx:02d}_final_{task_name}.jpg`

### Rollout Videos
Each episode generates an MP4 video saved in:
```
./rollouts/{date}/{timestamp}--episode={episode_idx}--success={True/False}--task={task_description}.mp4
```
