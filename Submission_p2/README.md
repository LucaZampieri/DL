# Project 2 of Deep-Learning
Course given by Francois Fleuret, EPFL EE-559
Implementation of a neural network from scratch.

## Folder organisation

##### Executables
- test.py is the main executable. Run it to get the results of our implementation of a simple net.
- pyTorchImplementation.py give the results for the same architecture than with test.py but implemented using pyTorch modules

##### Helpers
- modules.py contains our implementation of the modules used, including layers and losses
- optim.py contains our implementation of some optimizers
- loadData.py contains the function to generate, visualize and put in a suitable format the training, validation and test datasets
- tranAndTest.py contains the function to train the model, evaluate it and plot the training curves
- baseline.py contains the code that establishes some baselines with a linear classifier and data-augmented with the sum of the square of the coordinates (since we know the dataset is a circle, this will help the linear classifier to get good results)

## Authors
- Antoine Alleon, CSE student at EPFL
- Giogia Dellaferra, Physics student at EPFL
- Luca Zampieri, CSE student at EPFL
