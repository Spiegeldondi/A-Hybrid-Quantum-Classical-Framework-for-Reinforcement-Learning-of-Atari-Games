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

## Acknowledgments

This project incorporates code from the following sources:

- **[ageron/handson-ml2](https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb)**: The repository accompanying the book *[Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)* by Aurélien Géron (O’Reilly, 2019). © 2019 Aurélien Géron, ISBN 978-1-492-03264-9.

- **[TensorFlow Quantum tutorial on quantum reinforcement learning](https://github.com/tensorflow/quantum/blob/master/docs/tutorials/quantum_reinforcement_learning.ipynb)**: Licensed under the Apache License 2.0.

