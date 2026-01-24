# Neuron PWS Estimation Results

This directory contains Path Weight Sampling (PWS) mutual information estimates for individual neuron models.

## Contents

Each file `neuron_<id>_results.json` contains PWS estimation results for the corresponding neuron model from `models/neuron_<id>.pth`.

## Data Format

Each JSON file contains:

```json
{
  "neuron_id": <int>,
  "model_path": "<path to model checkpoint>",
  "args": {
    "n_neurons": 1,
    "hidden_size": 4,
    "num_layers": 2,
    "model_type": "CNN",
    "kernel_size": 20,
    "N": 400,        // Number of trajectory samples
    "M": 2048,       // Number of particles for marginal estimation
    "seq_len": 100   // Sequence length in time bins
  },
  "pws_result": {
    "t": [...],                        // Time grid
    "log_conditional": [...],          // Mean log p(x|s)
    "log_conditional_std": [...],      // Std dev of log p(x|s)
    "log_marginal": [...],             // Mean log p(x)
    "log_marginal_std": [...],         // Std dev of log p(x)
    "mutual_information": [...],       // Mean MI estimates
    "mutual_information_std": [...]    // Std dev of MI estimates
  }
}
```

## Generation

Results are generated using:
```bash
scripts/neurons/run_pws_all_neurons.sh
```

Or for individual neurons:
```bash
python scripts/neurons/estimate_pws.py models/neuron_<id>.pth \
    --output experiments/neurons_pws/neuron_<id>_results.json \
    --N 400 --M 2048
```

## Parameters

- **N = 400**: Number of stimulus-response trajectory pairs sampled from the stochastic harmonic oscillator
- **M = 2048**: Number of particles used for marginal likelihood estimation via SMC
- **seq_len = 100**: Trajectory length in 100ms time bins (10 seconds total)
- **Model**: CNN with kernel_size=20, 2 layers, hidden_size=4 per neuron

## Dataset

- Models were trained on salamander retinal ganglion cell spike responses to visual stimuli
- PWS estimation uses the stochastic harmonic oscillator as the stimulus model
- Results include both mean and standard deviation across N trajectory samples

## Validation Plots

Model validation plots showing predicted vs actual spike responses for each neuron for the validation stimulus. The top half of each panel shows the stimulus and the bottom half shows the response (blue: model prediction, orange: experimental data).

| Neuron 0 | Neuron 1 | Neuron 2 |
|----------|----------|----------|
| ![0](neuron_0/validation.png) | ![1](neuron_1/validation.png) | ![2](neuron_2/validation.png) |
| ![3](neuron_3/validation.png) | ![4](neuron_4/validation.png) | ![5](neuron_5/validation.png) |
| ![6](neuron_6/validation.png) | ![7](neuron_7/validation.png) | ![8](neuron_8/validation.png) |
| ![9](neuron_9/validation.png) | ![10](neuron_10/validation.png) | ![11](neuron_11/validation.png) |
| ![12](neuron_12/validation.png) | ![13](neuron_13/validation.png) | ![14](neuron_14/validation.png) |
| ![15](neuron_15/validation.png) | ![16](neuron_16/validation.png) | ![17](neuron_17/validation.png) |
| ![18](neuron_18/validation.png) | ![19](neuron_19/validation.png) | ![20](neuron_20/validation.png) |
| ![21](neuron_21/validation.png) | ![22](neuron_22/validation.png) | ![23](neuron_23/validation.png) |
| ![24](neuron_24/validation.png) | ![25](neuron_25/validation.png) | ![26](neuron_26/validation.png) |
| ![27](neuron_27/validation.png) | ![28](neuron_28/validation.png) | ![29](neuron_29/validation.png) |
| ![30](neuron_30/validation.png) | ![31](neuron_31/validation.png) | ![32](neuron_32/validation.png) |
| ![33](neuron_33/validation.png) | ![34](neuron_34/validation.png) | ![35](neuron_35/validation.png) |
| ![36](neuron_36/validation.png) | ![37](neuron_37/validation.png) | ![38](neuron_38/validation.png) |
| ![39](neuron_39/validation.png) | ![40](neuron_40/validation.png) | ![41](neuron_41/validation.png) |
| ![42](neuron_42/validation.png) | ![43](neuron_43/validation.png) | ![44](neuron_44/validation.png) |
| ![45](neuron_45/validation.png) | ![46](neuron_46/validation.png) | ![47](neuron_47/validation.png) |
| ![48](neuron_48/validation.png) | ![49](neuron_49/validation.png) | ![50](neuron_50/validation.png) |
| ![51](neuron_51/validation.png) | ![52](neuron_52/validation.png) | ![53](neuron_53/validation.png) |
| ![54](neuron_54/validation.png) | ![55](neuron_55/validation.png) | ![56](neuron_56/validation.png) |
| ![57](neuron_57/validation.png) | ![58](neuron_58/validation.png) | ![59](neuron_59/validation.png) |
| ![60](neuron_60/validation.png) | ![61](neuron_61/validation.png) | ![62](neuron_62/validation.png) |
| ![63](neuron_63/validation.png) | ![64](neuron_64/validation.png) | ![65](neuron_65/validation.png) |
| ![66](neuron_66/validation.png) | ![67](neuron_67/validation.png) | ![68](neuron_68/validation.png) |
| ![69](neuron_69/validation.png) | ![70](neuron_70/validation.png) | ![71](neuron_71/validation.png) |
| ![72](neuron_72/validation.png) | ![73](neuron_73/validation.png) | ![74](neuron_74/validation.png) |
| ![75](neuron_75/validation.png) | ![76](neuron_76/validation.png) | ![77](neuron_77/validation.png) |
| ![78](neuron_78/validation.png) | ![79](neuron_79/validation.png) | ![80](neuron_80/validation.png) |
| ![81](neuron_81/validation.png) | ![82](neuron_82/validation.png) | ![83](neuron_83/validation.png) |
| ![84](neuron_84/validation.png) | ![85](neuron_85/validation.png) | ![86](neuron_86/validation.png) |
| ![87](neuron_87/validation.png) | ![88](neuron_88/validation.png) | ![89](neuron_89/validation.png) |
| ![90](neuron_90/validation.png) | ![91](neuron_91/validation.png) | ![92](neuron_92/validation.png) |
| ![93](neuron_93/validation.png) | ![94](neuron_94/validation.png) | ![95](neuron_95/validation.png) |
| ![96](neuron_96/validation.png) | ![97](neuron_97/validation.png) | ![98](neuron_98/validation.png) |
| ![99](neuron_99/validation.png) | ![100](neuron_100/validation.png) | ![101](neuron_101/validation.png) |
| ![102](neuron_102/validation.png) | ![103](neuron_103/validation.png) | ![104](neuron_104/validation.png) |
| ![105](neuron_105/validation.png) | ![106](neuron_106/validation.png) | ![107](neuron_107/validation.png) |
| ![108](neuron_108/validation.png) | ![109](neuron_109/validation.png) | ![110](neuron_110/validation.png) |
| ![111](neuron_111/validation.png) | ![112](neuron_112/validation.png) | ![113](neuron_113/validation.png) |
| ![114](neuron_114/validation.png) | | |
