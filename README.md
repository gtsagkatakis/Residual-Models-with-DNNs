# Quantifying the structure of strong gravitational lens potentials with uncertainty aware deep neural networks

This repository contains code accompaning our paper:

Vernardos, G., Tsagkatakis, G., & Pantazis, Y. (2020). 
[Quantifying the structure of strong gravitational lens potentials with uncertainty-aware deep neural networks](https://doi.org/10.1093/mnras/staa3201). 
Monthly Notices of the Royal Astronomical Society, 499(4), 5641-5652.

# Abstract
Gravitational lensing is a powerful tool for constraining substructure in the mass distribution of galaxies, be it from the presence of dark matter sub-haloes or due to physical mechanisms affecting the baryons throughout galaxy evolution. Such substructure is hard to model and is either ignored by traditional, smooth modelling, approaches, or treated as well-localized massive perturbers. In this work, we propose a deep learning approach to quantify the statistical properties of such perturbations directly from images, where only the extended lensed source features within a mask are considered, without the need of any lens modelling. Our training data consist of mock lensed images assuming perturbing Gaussian Random Fields permeating the smooth overall lens potential, and, for the first time, using images of real galaxies as the lensed source. We employ a novel deep neural network that can handle arbitrary uncertainty intervals associated with the training data set labels as input, provides probability distributions as output, and adopts a composite loss function. The method succeeds not only in accurately estimating the actual parameter values, but also reduces the predicted confidence intervals by 10 per cent in an unsupervised manner, i.e. without having access to the actual ground truth values. Our results are invariant to the inherent degeneracy between mass perturbations in the lens and complex brightness profiles for the source. Hence, we can quantitatively and robustly quantify the smoothness of the mass density of thousands of lenses, including confidence intervals, and provide a consistent ranking for follow-up science.

# Description of files
The attached codes can be used for generating training/validataion data and executing training/inference

*Create_AnalysisReadyDataset.m*: Matlab code that loads the data (fits files) and produces the analysis ready data for training/validation

*Residual_modeling_with_DL.py*: Python/Tensorflow code for training/validating the deep learning models

*Weights.hdf5*: Saved weights

[residuals_July2019.zip](https://drive.google.com/file/d/1pgMPfPJoTv6v6-YSw5gE0ol5c637rIPZ/view?usp=sharing): Training/validataion data

# Depedencies
* Matlab 2019a
* Python 3.7.9
* Tensorflow 2.1

# Acknowledgements
G. Vernardos was funded by the GLADIUS project (contract no. 897124) within the H2020 Framework Program of the European Commission

G. Tsagkatakis was funded by the CALCHAS project (contract no. 842560) within the H2020 Framework Program of the European Commission
