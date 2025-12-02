# VLA Risk Evaluation

This project provides three scene types for evaluating VLA models on various datasets, supporting instructions and image attacks.

## Table of Contents

- [Script Overview](#script-overview)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Usage](#usage)
  - [Custom Instruction Evaluation](#custom-instruction-evaluation)
  - [Custom Image Evaluation (FLUX)](#custom-image-evaluation-flux)
- [Parameter Reference](#parameter-reference)
- [Output Description](#output-description)
- [Example Commands](#example-commands)

---

## Script Overview

### 1. `run_single_task_custom_instruction.py`
Evaluates model performance with **instructions**. Allows replacing the original task description with carefully crafted non-executable instructions.


### 2. `run_single_task_image_flux.py`
Evaluates model performance with **images**. Uses FLUX.1-Kontext model to naturally add text labels on images.

---

<table>
<tr>
<td width="50%">

**Custom Instruction Attack**

<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/ryc2001/VLA-Risk/main/case/true.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="50%">

**Image Attack with FLUX**

<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/ryc2001/VLA-Risk/main/case/false.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
</tr>
</table>

---

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

### Custom Image Evaluation

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

## Output

### Image Outputs
- **Intermediate images**: Saved in `{local_log_dir}/images_with_text/task_{task_id:02d}_{task_name}/episode_{episode_idx:02d}/step_{step:03d}.jpg`
- **Final images** (only for `run_single_task_custom_instruction.py`): Saved in `{images_dir}/episode_{episode_idx:02d}_final_{task_name}.jpg`

### Rollout Videos
Each episode generates an MP4 video saved in:
```
./rollouts/{date}/{timestamp}--episode={episode_idx}--success={True/False}--task={task_description}.mp4
```
