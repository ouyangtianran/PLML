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

# Acknowledgement

Some codes in this repository are borrowed from the following open-source projects. We extend our gratitude to the authors for their valuable contributions:

- [EP](https://github.com/ServiceNow/embedding-propagation)
- [MarginMatch](https://github.com/tsosea2/MarginMatch)
- [TorchSSL](https://github.com/TorchSSL/TorchSSL)
