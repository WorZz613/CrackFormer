# CrackFormer

CrackFormer is a deep learning-based semantic segmentation model designed for accurate and robust crack detection in challenging engineering scenarios, with a particular emphasis on shale fracture segmentation. The model captures fine-grained, irregular crack structures that are often difficult to identify using conventional methods or generic segmentation architectures.

This repository contains the official implementation of CrackFormer, as described in our paper submitted to *Advanced Engineering Informatics*. Model weights will be open-sourced upon publication of the paper.

**Key Features:**
- Tailored for shale crack segmentation under complex backgrounds and low-contrast conditions
- Effective at capturing multi-scale and elongated crack topologies
- Evaluation metrics including IoU, Dice, F1-score, and others

**Repository Structure:**
- `src/` – CrackFormer architecture
- `train.py` – training script
- `val.py` – validation and evaluation script
- `predict.py` – inference script for single images or batches
- `plot.py` – visualization and plotting utilities
- `utils/` – helper functions and utilities
- `log/` – training logs and output records
- `requirements.txt` – dependency list

**Citation:**
If you find CrackFormer useful, please cite our paper (citation details will be added upon publication).****
