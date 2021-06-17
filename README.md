# Quantifying the structure of strong gravitational lens potentials with uncertainty aware deep neural networks

This repository contains code accompaning our paper:

Vernardos, G., Tsagkatakis, G., & Pantazis, Y. (2020). 
[Quantifying the structure of strong gravitational lens potentials with uncertainty-aware deep neural networks](https://doi.org/10.1093/mnras/staa3201). 
Monthly Notices of the Royal Astronomical Society, 499(4), 5641-5652.

# Description of files
The attached codes can be used for generating training/validataion data and executing training/inference

*Create_AnalysisReadyDataset.m*: Matlab code that loads the data (fits files) and produces the analysis ready date for training/validation

*Residual_modeling_with_DL.py*: Python/Tensorflow code for training/validating the deep learning models

*Weights.hdf5*: Saved weights

[residuals_July2019.zip](https://drive.google.com/file/d/1pgMPfPJoTv6v6-YSw5gE0ol5c637rIPZ/view?usp=sharing): Training/validataion data

# Depedencies
* Matlab 2019a
* Python 3.7.9
* Tensorflow 2.1
