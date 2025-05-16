# Pseudo-Labeling Based Practical Semi-Supervised Meta-Training for Few-Shot Learning

This repository contains the official code for "Pseudo-Labeling Based Practical Semi-Supervised Meta-Training for Few-Shot Learning".

## Usage

### Step 1: Data Preparation

Download the required data and pre-trained models from [this link](https://drive.google.com/file/d/1ddxNe1slXSFVkF3pli2WP0x6JRwYYdOg/view?usp=sharing) and place the `data/` folder inside `EP-semi/`. If necessary, you can also use `EP-semi/get_split_index_for_semi_v2.py` to re-split the data.

### Step 2: Semi-Supervised Pre-training

1. Set up the environment by running `install.sh` inside `MarginMatch/`.
2. Run `mini_conv4.sh` inside `MarginMatch/script`.

**Note**:
- You can modify the relevant configurations in `MarginMatch/config/marginmatch/marginmatch_mini_050.yaml`.
- If you don't want to retrain, you can directly place the provided models in the `MarginMatch/saved_models/` folder.

### Step 3: PLML (Pseudo-Labeling Meta-Learning)

1. Set up the environment by running `create_env.sh` inside `EP-semi/`.
2. Run `mini_marginmatch.sh` inside `EP-semi/scripts_noise/`.
3. Then you can check the results in `EP-semi/logs/`.

**Note**:
- You can modify the relevant configurations in `EP-semi/exp_configs_emb_semi/finetune_exps.py`.

We apologize for the incomplete organization of the code due to our busy schedule. The remaining code will be released once it is fully organized. In the meantime, feel free to tweak the available code according to your needs or contact us for assistance.

---

## Experimental Results

> ★ = 1-shot ✦ = 5-shot **Bold** = best in each block  

### miniImageNet · Conv-4 · in-Proto

| Pre-train | Fine-tune | Inference | 20★ | 50★ | 100★ | Full★ | 20✦ | 50✦ | 100✦ | Full✦ |
|-----------|-----------|-----------|-----:|-----:|------:|------:|-----:|-----:|------:|------:|
| Base | – | in-Proto | 39.86 | 43.37 | 45.53 | 48.64 | 55.09 | 60.26 | 63.22 | 67.45 |
| Base | Proto | in-Proto | 38.58 | 43.37 | 44.89 | **49.77** | 54.29 | 59.62 | 62.36 | **68.58** |
| SemCo | – | in-Proto | 42.68 | 44.98 | 47.01 | – | 61.71 | 63.55 | 65.71 | – |
| **PLML-Proto** | SemCo | in-Proto | **45.24** | **45.13** | **47.57** | – | **64.94** | **66.16** | **68.32** | – |
| FlexMatch | – | in-Proto | 40.57 | 42.35 | 44.91 | – | 59.01 | 61.69 | 62.92 | – |
| **PLML-Proto** | FlexMatch | in-Proto | 42.55 | 43.75 | 46.63 | – | 62.83 | 63.62 | 67.01 | – |
| MarginMatch | – | in-Proto | 41.46 | 41.10 | 44.23 | – | 58.93 | 60.28 | 62.64 | – |
| **PLML-Proto** | MarginMatch | in-Proto | 42.86 | 44.47 | 46.93 | – | 62.87 | 64.49 | 66.75 | – |

### miniImageNet · Conv-4 · trans-EP

| Pre-train | Fine-tune | Inference | 20★ | 50★ | 100★ | Full★ | 20✦ | 50✦ | 100✦ | Full✦ |
|-----------|-----------|-----------|-----:|-----:|------:|------:|-----:|-----:|------:|------:|
| Base | – | trans-EP | 43.33 | 47.01 | 49.54 | 53.06 | 55.92 | 60.36 | 63.74 | 67.23 |
| Base | EP | trans-EP | 42.31 | 46.78 | 49.01 | **54.99** | 54.79 | 61.07 | 63.60 | **68.92** |
| SemCo | – | trans-EP | 47.56 | 49.44 | 52.10 | – | 61.39 | 64.18 | 66.57 | – |
| **PLML-EP** | SemCo | trans-EP | **52.59** | **54.03** | **56.23** | – | **66.30** | **67.92** | **70.05** | – |
| FlexMatch | – | trans-EP | 45.35 | 47.77 | 49.57 | – | 58.75 | 61.76 | 64.02 | – |
| **PLML-EP** | FlexMatch | trans-EP | 50.59 | 52.22 | 55.14 | – | 64.47 | 65.29 | 68.78 | – |
| MarginMatch | – | trans-EP | 45.17 | 48.32 | 49.83 | – | 59.48 | 60.99 | 63.13 | – |
| **PLML-EP** | MarginMatch | trans-EP | 49.93 | 52.13 | 54.39 | – | 63.81 | 65.88 | 67.85 | – |

### miniImageNet · ResNet-12 · in-Proto

| Pre-train | Fine-tune | Inference | 20★ | 50★ | 100★ | Full★ | 20✦ | 50✦ | 100✦ | Full✦ |
|-----------|-----------|-----------|-----:|-----:|------:|------:|-----:|-----:|------:|------:|
| Base | – | in-Proto | 37.06 | 41.41 | 45.37 | 51.05 | 55.95 | 63.50 | 68.70 | 75.97 |
| Base | Proto | in-Proto | 34.97 | 41.13 | 43.43 | **54.20** | 54.71 | 62.95 | 66.32 | **76.21** |
| SemCo | – | in-Proto | 46.52 | 46.82 | 47.35 | – | 72.27 | 73.38 | 73.96 | – |
| **PLML-Proto** | SemCo | in-Proto | **49.74** | **49.21** | **51.15** | – | **76.35** | **76.25** | **76.50** | – |
| FlexMatch | – | in-Proto | 45.40 | 46.30 | 45.52 | – | 70.55 | 71.85 | 70.79 | – |
| **PLML-Proto** | FlexMatch | in-Proto | 47.03 | 47.54 | 48.09 | – | 72.46 | 73.20 | 75.03 | – |
| MarginMatch | – | in-Proto | 45.84 | 45.71 | 44.36 | – | 68.31 | 70.44 | 69.08 | – |
| **PLML-Proto** | MarginMatch | in-Proto | **50.15** | **50.84** | **51.66** | – | 71.96 | 73.22 | 75.09 | – |

### miniImageNet · ResNet-12 · trans-EP

| Pre-train | Fine-tune | Inference | 20★ | 50★ | 100★ | Full★ | 20✦ | 50✦ | 100✦ | Full✦ |
|-----------|-----------|-----------|-----:|-----:|------:|------:|-----:|-----:|------:|------:|
| Base | – | trans-EP | 41.54 | 48.09 | 52.43 | 60.05 | 54.93 | 63.02 | 68.65 | 76.28 |
| Base | EP | trans-EP | 44.77 | 50.63 | 54.93 | **65.75** | 58.13 | 65.72 | 69.72 | **79.16** |
| SemCo | – | trans-EP | 60.34 | 61.57 | 62.32 | – | 75.10 | 76.11 | 76.47 | – |
| **PLML-EP** | SemCo | trans-EP | **64.45** | **64.78** | 64.40 | – | **77.67** | 78.00 | **78.84** | – |
| FlexMatch | – | trans-EP | 52.50 | 54.20 | 52.75 | – | 69.53 | 71.13 | 69.70 | – |
| **PLML-EP** | FlexMatch | trans-EP | 63.01 | 64.39 | **64.52** | – | 76.96 | **78.29** | 78.65 | – |
| MarginMatch | – | trans-EP | 55.86 | 57.89 | 58.17 | – | 70.19 | 73.10 | 73.28 | – |
| **PLML-EP** | MarginMatch | trans-EP | 59.55 | 61.26 | 62.86 | – | 73.84 | 75.91 | 77.13 | – |

---

### tieredImageNet · Conv-4 · in-Proto

| Pre-train | Fine-tune | Inference | 20★ | 100★ | 200★ | All★ | 20✦ | 100✦ | 200✦ | All✦ |
|-----------|-----------|-----------|-----:|------:|------:|-----:|-----:|------:|------:|-----:|
| Base | Proto | in-Proto | **45.61** | 50.80 | 51.07 | **52.50** | 62.43 | 68.48 | 70.12 | **71.50** |
| **PLML-Proto** | SemCo | in-Proto | 45.17 | **51.86** | **52.04** | – | 64.25 | **72.43** | **72.64** | – |
| **PLML-Proto** | FlexMatch | in-Proto | 44.94 | 49.33 | 50.60 | – | **64.80** | 70.54 | 71.32 | – |

### tieredImageNet · Conv-4 · trans-EP

| Pre-train | Fine-tune | Inference | 20★ | 100★ | 200★ | All★ | 20✦ | 100✦ | 200✦ | All✦ |
|-----------|-----------|-----------|-----:|------:|------:|-----:|-----:|------:|------:|-----:|
| Base | EP | trans-EP | 49.33 | 55.53 | 56.08 | **58.10** | 62.71 | 68.67 | 70.01 | **71.44** |
| **PLML-EP** | SemCo | trans-EP | 52.45 | **60.52** | **61.09** | – | 65.49 | **73.90** | **74.29** | – |
| **PLML-EP** | FlexMatch | trans-EP | **54.41** | 58.88 | 60.56 | – | **68.43** | 72.71 | 73.54 | – |

### tieredImageNet · Conv-4 · semi-EP

| Pre-train | Fine-tune | Inference | 20★ | 100★ | 200★ | All★ | 20✦ | 100✦ | 200✦ | All✦ |
|-----------|-----------|-----------|-----:|------:|------:|-----:|-----:|------:|------:|-----:|
| Base | EP | semi-EP | 54.02 | 61.19 | 61.96 | **63.28** | 65.03 | 70.87 | 71.98 | **74.14** |
| **PLML-EP** | SemCo | semi-EP | 56.85 | **67.71** | **68.20** | – | 68.20 | **75.74** | **76.66** | – |
| **PLML-EP** | FlexMatch | semi-EP | **60.50** | 65.93 | 66.37 | – | **71.38** | 74.85 | 75.28 | – |

### tieredImageNet · ResNet-12 · in-Proto

| Pre-train | Fine-tune | Inference | 20★ | 100★ | 200★ | All★ | 20✦ | 100✦ | 200✦ | All✦ |
|-----------|-----------|-----------|-----:|------:|------:|-----:|-----:|------:|------:|-----:|
| Base | Proto | in-Proto | 44.59 | **56.03** | **59.73** | **64.24** | 68.91 | 78.27 | 80.90 | **83.66** |
| **PLML-Proto** | SemCo | in-Proto | 49.55 | 55.35 | 57.41 | – | 74.79 | **80.48** | **82.17** | – |
| **PLML-Proto** | FlexMatch | in-Proto | **51.60** | 54.12 | 58.25 | – | **75.97** | 79.66 | 81.61 | – |

### tieredImageNet · ResNet-12 · trans-EP

| Pre-train | Fine-tune | Inference | 20★ | 100★ | 200★ | All★ | 20✦ | 100✦ | 200✦ | All✦ |
|-----------|-----------|-----------|-----:|------:|------:|-----:|-----:|------:|------:|-----:|
| Base | EP | trans-EP | 55.68 | 66.67 | 68.40 | **72.66** | 71.19 | 80.79 | 80.96 | **84.59** |
| **PLML-EP** | SemCo | trans-EP | **68.08** | **70.69** | **73.51** | – | **81.08** | **83.09** | **84.75** | – |
| **PLML-EP** | FlexMatch | trans-EP | 66.65 | 69.98 | 71.81 | – | 79.29 | 82.53 | 83.66 | – |

### tieredImageNet · ResNet-12 · semi-EP

| Pre-train | Fine-tune | Inference | 20★ | 100★ | 200★ | All★ | 20✦ | 100✦ | 200✦ | All✦ |
|-----------|-----------|-----------|-----:|------:|------:|-----:|-----:|------:|------:|-----:|
| Base | EP | semi-EP | 65.15 | 74.62 | 75.52 | **79.97** | 75.60 | 83.10 | 83.54 | **86.61** |
| **PLML-EP** | SemCo | semi-EP | **75.13** | 76.87 | **79.62** | – | **82.42** | **85.07** | **86.69** | – |
| **PLML-EP** | FlexMatch | semi-EP | 73.93 | **77.02** | 79.12 | – | 81.55 | 84.16 | 85.88 | – |

---

# Acknowledgement

Some codes in this repository are borrowed from the following open-source projects. We extend our gratitude to the authors for their valuable contributions:

- [EP](https://github.com/ServiceNow/embedding-propagation)
- [MarginMatch](https://github.com/tsosea2/MarginMatch)
- [TorchSSL](https://github.com/TorchSSL/TorchSSL)
