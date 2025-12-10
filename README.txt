A small, clean PyTorch implementation of a simple convolutional neural network (CNN) trained on the MNIST handwritten digit dataset.
The goal of this project is to provide a minimal but well-structured example of model definition, training, evaluation, and data handling.

Features:
- Implementation from scratch in PyTorch 
- Achieves ~93% accuracy on the MINST test set
- Modular codebase (separate model and data modules)
- Includes basic training loop, evaluation, and learning curves

Files:
network.py : CNN model definition
data.py : dataset loading utilities (train/test split, transforms)
main.py : training script (runs training + evaluation, plots learning curve)