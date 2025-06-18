# Cost-Accuracy Trade-off on a simple Derivative task

## Problem Overview
- The project aims at exploring the cost-accuracy trade-off in operator learning, inspired by the paper *"The Cost-Accuracy Trade-off in Operator Learning with Neural Networks."*
- Considering functions constructed from sine and cosine basis:
```math
          u(x) = p_{sin} * sin(x) + p_{cos} * cos(x)
```
- The goal is to learn an operator that maps $u(x)$ to its derivative:
```math
v(x)=\frac{du}{dx} = 1_{\sin} \cdot \sin(x) +q_{\cos} \cdot \cos(x)
```
- We implement and compare **three neural operator architectures**: 
  - PCA-Net
  - DeepONet
  - PARA-Net

## Project structure
### Data Generation
- `data_generation.py`
  - Goal: Generate synthetic traning and testing data (`u`, `v`)
  - They are saved in `data/train/` and `data/test/`

### Model training files
- Generally, each script:
  - Loading training/ test data
  - Training the model with different **network widths** (`width = 16, 32, 64, 126, 256`)
  - Evaluateing **test error** and **Flops**
  - Saving the result in a `.txt` file
- The files are named as:
  - `train_pca_net.py`
  - `train_deep_onet.py`
  - `train_para_net.py`

### Evaluation cost
`evaluate_flops.py`
- Goal: Store the FLOP expression for each architecture

### Plotting
`cost_accuracy_tradeoff_plot.py`
- Reading test errors and FLOPs 
- Plotting test error versus FLOP








