# A-Hybrid-Quantum-Classical-Framework-for-Reinforcement-Learning-of-Atari-Games
This repo explores the capabilities of hybrid quantum-classical models based on variational quantum algorithms (VQAs) in high dimensional reinforcement learning (RL) environments such as the Atari 2600 classics Pong and Breakout.

---

## Installation Guide

Follow these steps to set up the project on an Ubuntu-based system.

### 1. System Requirements
- **Operating System**: Ubuntu 20.04+ (or equivalent Linux distribution)
- **Python Version**: Python 3.8 (required for compatibility)

---

### 2. Install System Dependencies
```bash
sudo apt update
sudo apt install -y software-properties-common
```

---

### 3. Install Python 3.8 and Virtual Environment Tools
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.8 python3.8-venv python3.8-distutils
```

---

### 4. Create and Activate a Virtual Environment
```bash
python3.8 -m venv myenv
source myenv/bin/activate
```

---

### 5. Install Build Tools
```bash
sudo apt install -y cmake make gcc g++ python3.8-dev
```

---

### 6. Install Required Libraries
```bash
sudo apt install -y zlib1g-dev libgl1
```

---

### 7. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ASDF

### How to run
To execute the project, use the following command in your terminal:
```bash
python breakout-v5.py [model] [activation] [lr1] [lr2] [lr3] [lr4] [lr5] [--lr6 LR6] --n_qubits N_QUBITS --n_layers N_LAYERS --scaling SCALING --id ID --seed SEED [--bottleneck {0,1}] [--path PATH]
```
or for Pong:
```bash
python pong-v3.py [model] [activation] [lr1] [lr2] [lr3] [lr4] [lr5] [--lr6 LR6] --n_qubits N_QUBITS --n_layers N_LAYERS --scaling SCALING --id ID --seed SEED [--bottleneck {0,1}] [--path PATH]
```
The following table explains the arguments.

### Arguments Overview

| Argument            | Description                                           | Required |
|---------------------|-------------------------------------------------------|----------|
| `model`             | Model type: `"classic"` or `"quantum"`                | Yes      |
| `activation`        | Activation function in pre-processing layer: `"linear"`, `"tanh"`, or `"relu"` | Yes      |
| `lr1` - `lr5`       | Learning rates for the respective layers              | Yes      |
| `--lr6`             | Learning rate for the post-processing layer (required if `model=quantum`) | No       |
| `--n_qubits`        | Number of qubits in the PQC                          | Yes      |
| `--n_layers`        | Number of layers in the PQC                         | Yes      |
| `--scaling`         | Scaling applied to rewards                          | Yes      |
| `--id`              | Unique 2-digit ID                                    | Yes      |
| `--seed`            | Random seed                                          | Yes      |
| `--bottleneck`      | Include bottleneck layer (required if `model=classic`) | No       |
| `--path`            | Path for saving output files (default: current directory) | No       |

---

## Acknowledgments

This project incorporates code from the following sources:

- **[ageron/handson-ml2](https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb)**: The repository accompanying the book *[Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)* by Aurélien Géron (O’Reilly, 2019). © 2019 Aurélien Géron, ISBN 978-1-492-03264-9.

- **[TensorFlow Quantum tutorial on quantum reinforcement learning](https://github.com/tensorflow/quantum/blob/master/docs/tutorials/quantum_reinforcement_learning.ipynb)**: Licensed under the Apache License 2.0.

