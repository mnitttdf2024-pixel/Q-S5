# Bin Directory Documentation

The `bin/` directory contains **48 shell scripts** that orchestrate training, evaluation, and quantization experiments for the Q-S5 (Quantized State Space Model) project. These are not binary files -- they are Bash scripts designed to be submitted as SLURM jobs on an HPC GPU cluster.

Additionally, three shell scripts in the repository root (`install_deps.sh`, `qaa_all.sh`, `run_scifar_all copy.sh`) serve as top-level helpers.

---

## Table of Contents

1. [How the Scripts Work](#how-the-scripts-work)
2. [Configuration Parameters Reference](#configuration-parameters-reference)
3. [Model Architecture Parameters](#model-architecture-parameters)
4. [Training Parameters](#training-parameters)
5. [Infrastructure / Logging Parameters](#infrastructure--logging-parameters)
6. [Quantization Naming Conventions](#quantization-naming-conventions)
7. [Script Categories](#script-categories)
8. [Root-Level Scripts](#root-level-scripts)
9. [Per-Dataset Configurations](#per-dataset-configurations)

---

## How the Scripts Work

Each script in `bin/` follows a common pattern:

```
#!/bin/bash
#SBATCH directives       # SLURM cluster scheduling
[argument parsing]       # Accept quantization overrides from the command line
[environment setup]      # Pyenv, virtualenv, WANDB API key
python run_qtrain.py … # Launch the actual training/eval Python process
```

Scripts are submitted to the cluster via `sbatch`:

```bash
sbatch bin/run_scifar.sh --a_bits=8 --b_bits=8 --c_bits=8 --d_bits=8 --non_ssm_bits=8
```

### SLURM Directives

| Directive | Meaning |
|---|---|
| `#SBATCH -p gpu` (or `-p g24`, `-p g80`) | GPU partition to use |
| `#SBATCH --gres=gpu:1` | Request 1 GPU |
| `#SBATCH -c 14` | Request 14 CPU cores |
| `#SBATCH -t 10:00:00` / `24:00:00` | Wall-clock time limit |
| `#SBATCH --qos=high` | Quality-of-service priority (some scripts) |
| `#SBATCH -o .../slurm-%j.out` | SLURM log output path |

---

## Configuration Parameters Reference

These are the quantization-specific parameters that can be passed to the bin scripts. They control how many bits are used to represent different weight matrices and activations in the S5 model.

### Quantization Bit-Width Parameters

| Parameter | What It Controls | Typical Values | Notes |
|---|---|---|---|
| `--a_bits` | Bit width for the **A matrix** (state transition matrix) | 2, 4, 8, or `None` (full precision) | The A matrix linearly maps the recurrent state over time. It is the **most sensitive** component to quantization -- the paper shows it degrades below 8 bits on many LRA tasks. |
| `--b_bits` | Bit width for the **B matrix** (input-to-state mapping) | 2, 4, 8 | Controls how the input signal is projected into the latent state. More robust to low-bit quantization than A. |
| `--c_bits` | Bit width for the **C matrix** (state-to-output mapping) | 2, 4, 8 | Controls how the latent state is projected back to the output. Similar robustness profile to B. |
| `--d_bits` | Bit width for the **D matrix** (skip/feedthrough connection) | 2, 4, 8 | A direct input-to-output connection that bypasses the recurrence. Typically grouped with B and C for quantization. |
| `--ssm_act_bits` | Bit width for **SSM activation values** | 8 | Activations (intermediate computation results) within the SSM layers. Almost always kept at 8 bits. |
| `--non_ssm_bits` | Bit width for **non-SSM weight matrices** | 2, 4, 8 | Weights in layers outside the SSM (encoder, decoder, MLPs). These can often tolerate aggressive quantization. |
| `--non_ssm_act_bits` | Bit width for **non-SSM activation values** | 8 | Activations in non-SSM layers. Almost always kept at 8 bits. |

When any `*_bits` parameter is set to `None` (the default), that component runs in full precision (float32).

### Activation Function Approximations

| Parameter | What It Controls | Default |
|---|---|---|
| `--qgelu_approx` | Use a **quantization-friendly GELU approximation** instead of exact GELU | `False` |
| `--hard_sigmoid` | Use a **hard sigmoid** approximation instead of exact sigmoid | `False` |

These approximations make the model more hardware-friendly for integer-only inference by replacing transcendental functions with piecewise-linear alternatives.

### Normalization Parameters

| Parameter | What It Controls | Default |
|---|---|---|
| `--batchnorm` | Use **BatchNorm** (`True`) or **LayerNorm** (`False`) | `True` |
| `--use_qlayernorm_if_quantized` | When quantized and using LayerNorm, use a **quantized LayerNorm** implementation | `True` (in eval scripts) |
| `--use_layernorm_bias` | Include a bias term in LayerNorm | `True` |
| `--remove_norm_bias_from_checkpoint` | Strip norm bias values when loading a checkpoint (for compatibility) | `False` |

---

## Model Architecture Parameters

These are hard-coded per dataset in each script and define the S5 model structure.

| Parameter | Symbol | Meaning |
|---|---|---|
| `--n_layers` | -- | Number of stacked S5 layers |
| `--d_model` | H | Dimensionality of input/output features |
| `--ssm_size_base` | P | Latent state size inside each SSM |
| `--blocks` | J | Number of blocks used to initialize the A matrix (block-diagonal structure) |
| `--C_init` | -- | Initialization strategy for the C matrix (`lecun_normal` or `trunc_standard_normal`) |
| `--bidirectional` | -- | Whether the SSM processes the sequence in both directions |
| `--prenorm` | -- | Apply normalization before (True) or after the SSM layer |
| `--clip_eigs` | -- | Clip eigenvalues of A to ensure stability |
| `--dt_global` | -- | Use a global discretization timestep (used in retrieval tasks) |

---

## Training Parameters

| Parameter | Meaning |
|---|---|
| `--epochs` | Number of training epochs |
| `--bsz` | Batch size |
| `--ssm_lr_base` | Base learning rate for SSM parameters |
| `--lr_factor` | Multiplier applied to the base LR for non-SSM parameters |
| `--p_dropout` | Dropout probability |
| `--weight_decay` | Weight decay coefficient |
| `--warmup_end` | Epoch at which learning rate warmup ends |
| `--opt_config` | Optimizer configuration preset (`standard`, `BfastandCdecay`, etc.) |
| `--jax_seed` | Random seed for reproducibility |

---

## Infrastructure / Logging Parameters

| Parameter | Meaning |
|---|---|
| `--USE_WANDB` | Enable Weights & Biases logging |
| `--wandb_project` | W&B project name |
| `--wandb_entity` | W&B team/entity |
| `--mlflow_tracking_uri` | MLflow tracking server URL |
| `--mlflow_experiment_id` | MLflow experiment identifier |
| `--mlflow_run_id` | MLflow run identifier (for resuming) |
| `--job_id` | SLURM job ID (passed as `$SLURM_JOB_ID`) |
| `--run_name` | Human-readable name for the experiment run |
| `--load_run_name` | Name of a previous run whose checkpoint to load (for eval/QAA) |
| `--checkpoint_dir` | Directory where model checkpoints are saved/loaded |

---

## Quantization Naming Conventions

The file names and run names use a consistent shorthand to describe quantization configurations:

| Shorthand | Meaning | Bit Assignments |
|---|---|---|
| **W8A8** | 8-bit weights, 8-bit activations | `a=8, b=8, c=8, d=8, non_ssm=8, ssm_act=8, non_ssm_act=8` |
| **W4A8** | 4-bit weights everywhere, 8-bit activations | `a=4, b=4, c=4, d=4, non_ssm=4, ssm_act=8, non_ssm_act=8` |
| **W4A8Wssm8** | 4-bit non-SSM weights, 8-bit SSM weights, 8-bit activations | `a=8, b=8, c=8, d=8, non_ssm=4, ssm_act=8, non_ssm_act=8` |
| **W4A8Wa8** | 4-bit weights with A matrix kept at 8 bits, 8-bit activations | `a=8, b=4, c=4, d=4, non_ssm=4, ssm_act=8, non_ssm_act=8` |
| **W2A8** | 2-bit weights everywhere, 8-bit activations | `a=2, b=2, c=2, d=2, non_ssm=2, ssm_act=8, non_ssm_act=8` |
| **W2A8Wssm8** | 2-bit non-SSM weights, 8-bit SSM weights, 8-bit activations | `a=8, b=8, c=8, d=8, non_ssm=2, ssm_act=8, non_ssm_act=8` |
| **W2A8Wa8** | 2-bit weights with A matrix kept at 8 bits, 8-bit activations | `a=8, b=2, c=2, d=2, non_ssm=2, ssm_act=8, non_ssm_act=8` |

Suffixes in filenames further specify activation function and norm variants:

| Suffix | Meaning |
|---|---|
| `-a8` | A matrix kept at 8 bits (same as `Wa8`) |
| `-ssm8` | All SSM matrices kept at 8 bits (same as `Wssm8`) |
| `qgelu` | Uses quantized GELU approximation |
| `hsigmoid` | Uses hard sigmoid approximation |
| `qln` / `qlayernorm` | Uses quantized LayerNorm |
| `ln` | Uses standard LayerNorm |
| `ln_nb` | LayerNorm with no bias |
| `lnnb` | LayerNorm no-bias (alternate abbreviation) |

---

## Script Categories

### 1. Training Scripts (`run_<dataset>.sh`)

Full-precision or quantized training from scratch. Invokes `run_qtrain.py`.

| Script | Dataset | Layers | d_model | SSM Size | Blocks |
|---|---|---|---|---|---|
| `run_smnist.sh` | MNIST classification | 4 | 96 | 128 | 1 |
| `run_scifar.sh` | LRA-CIFAR classification | 6 | 512 | 384 | 3 |
| `run_listops.sh` | ListOps classification | 8 | 128 | 256 | 8 |
| `run_text.sh` | IMDB text classification | 6 | 256 | 256 | 8 |
| `run_pathfinder.sh` | Pathfinder classification | 6 | 192 | 256 | 8 |
| `run_pathx.sh` | PathX classification | 6 | 384 | 512 | 16 |
| `run_retrieval.sh` | AAN retrieval classification | 6 | 128 | 256 | 16 |

### 2. Quantized Training Variants (`run_<dataset>_w<N>a<M>*.sh`)

Same as training scripts but with quantization bit widths hard-coded. Examples:

- `run_pathfinder_w8a8.sh` -- 8-bit weights and activations
- `run_pathfinder_w4a8.sh` -- 4-bit weights, 8-bit activations
- `run_pathfinder_w4a8-ssm8.sh` -- 4-bit non-SSM weights, SSM weights at 8-bit
- `run_pathfinder_w4a8-a8.sh` -- 4-bit weights with A matrix at 8-bit
- `run_pathfinder_w2a8.sh` -- 2-bit weights, 8-bit activations

The same naming pattern applies to PathX and Retrieval tasks.

### 3. Evaluation Scripts (`eval_<dataset>.sh`)

Load a trained checkpoint and evaluate it, optionally with post-training quantization applied. Invokes `run_qeval.py`. These support additional parameters like `--load_run_name` and `--use_qlayernorm_if_quantized`.

| Script | Dataset |
|---|---|
| `eval_smnist.sh` | MNIST |
| `eval_scifar.sh` | LRA-CIFAR |
| `eval_listops.sh` | ListOps |
| `eval_text.sh` | IMDB text |

### 4. Post-Training Quantization (PTQ) Batch Scripts (`run_ptq_*_eval*.sh`)

These are orchestrator scripts that call the eval scripts multiple times with different quantization configurations to sweep through PTQ experiments. They test combinations of:

- Different bit widths (W8A8 down to W2A8)
- Exact vs. approximate activation functions (GELU vs. qGELU, sigmoid vs. hard sigmoid)
- Standard vs. quantized LayerNorm

| Script | Dataset | Normalization |
|---|---|---|
| `run_ptq_smnist_eval.sh` | MNIST | BatchNorm |
| `run_ptq_smnist_eval_lnnb.sh` | MNIST | LayerNorm (no bias) |
| `run_ptq_scifar_eval_lnnb.sh` | LRA-CIFAR | LayerNorm (no bias) |
| `run_ptq_listops_eval_lnnb.sh` | ListOps | LayerNorm (no bias) |
| `run_ptq_text_eval_lnnb.sh` | IMDB text | LayerNorm (no bias) |

### 5. Quantization-Aware Accuracy (QAA) Scripts (`qaa_<dataset>.sh`)

Fine-tune a pre-trained full-precision model with quantization enabled (quantization-aware training/fine-tuning). These load a checkpoint via `--load_run_name`, apply quantization, and continue training for a reduced number of epochs with a lower learning rate.

| Script | Dataset | Fine-tune Epochs | SSM LR |
|---|---|---|---|
| `qaa_smnist.sh` | MNIST | 15 | 0.00001 |
| `qaa_scifar.sh` | LRA-CIFAR | 25 | 0.00001 |
| `qaa_listops.sh` | ListOps | 15 | 0.000005 |
| `qaa_text.sh` | IMDB text | 10 | 0.000005 |

### 6. Batch Runner Scripts (`run_<dataset>_all.sh`, `qaa_all.sh`)

These orchestrate multiple runs across different quantization configurations for a given dataset. They loop through configurations like W8A8, W4A8, W4A8Wssm8, W2A8, etc., submitting one SLURM job per configuration.

| Script | Purpose |
|---|---|
| `run_smnist_all.sh` | Run all MNIST quantization configs |
| `run_scifar_all.sh` | Run all CIFAR quantization configs |
| `run_listops_all.sh` | Run all ListOps quantization configs |
| `run_text_all.sh` | Run all text quantization configs |
| `run_retrieval_all.sh` | Run all retrieval quantization configs |
| `qaa_all.sh` (root) | Run QAA fine-tuning for all configs on a given dataset |

### 7. Special Variant Scripts

| Script | Purpose |
|---|---|
| `run_smnist_Alow_laynorm.sh` | MNIST with low A-matrix precision + LayerNorm |
| `run_smnist_all_laynorm.sh` | MNIST batch runs using LayerNorm instead of BatchNorm |
| `run_smnist_all_nqact.sh` | MNIST batch runs without quantized activations |
| `run_ret4a8-a8.sh` | Retrieval with W4A8 and A matrix at 8 bits |

---

## Root-Level Scripts

### `install_deps.sh`

Sets up the Python environment and installs all dependencies:
- Creates a Python 3.10.10 virtualenv
- Installs JAX with CUDA 12 support
- Installs the local AQT (Accurate Quantized Training) framework
- Installs experiment tracking (wandb), data loading (torchtext, datasets, tensorflow-datasets), and utility libraries (tqdm, einops)

### `qaa_all.sh`

A top-level batch runner for QAA experiments. Takes a dataset name as an argument and submits SLURM jobs for all 7 standard quantization configurations (W8A8 through W2A8). Uses hard sigmoid, quantized GELU, and quantized LayerNorm for all runs.

Usage:
```bash
./qaa_all.sh scifar
```

---

## Per-Dataset Configurations

Summary of the model architecture settings used per dataset (from the Long Range Arena benchmark):

| Dataset | Task | Seq Length | Layers | d_model | SSM Size | Blocks | Bidirectional | Optimizer | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | Image classification | 784 | 4 | 96 | 128 | 1 | No | standard | 150 |
| LRA-CIFAR | Image classification | 1024 | 6 | 512 | 384 | 3 | Yes | BfastandCdecay | 250 |
| ListOps | Sequence processing | 2048 | 8 | 128 | 256 | 8 | Yes | standard | 40 |
| IMDB Text | Text classification | 4096 | 6 | 256 | 256 | 8 | Yes | BfastandCdecay | 50 |
| Pathfinder | Visual reasoning | 1024 | 6 | 192 | 256 | 8 | Yes | standard | 200 |
| PathX | Extended visual reasoning | 16384 | 6 | 384 | 512 | 16 | Yes | standard | 75 |
| AAN Retrieval | Document retrieval | 4000 | 6 | 128 | 256 | 16 | Yes | standard | 20 |
