# HierarchicalDesign: A computational building block approach towards multiscale architected materials analysis and design with application to hierarchical metal metamaterials    
### Markus J. Buehler
email: mbuehler@mit.edu  

We report a computational approach towards multiscale architected materials analysis and design. A particular challenge in modeling and simulation of materials, and especially the development of hierarchical design approaches, has been to identify ways by which complex multi-level material structures can be effectively modeled. One way to achieve this is to use coarse-graining approaches, where  physical relationships can be effectively described with reduced dimensionality. In this paper we report an integrated deep neural network architecture that first learns coarse-grained representations of complex hierarchical microstructure data via a discrete variational autoencoder and then utilizes an attention-based diffusion model solve both forward and inverse problems, including a capacity to solve degenerate design problems. As an application, we demonstrate the method in the analysis and design of hierarchical highly porous metamaterials within the context of nonlinear stress-strain responses to compressive deformation.  We validate the mechanical behavior and mechanisms of deformation using embedded-atom molecular dynamics simulations carried out for copper and nickel, showing good agreement with the design objectives.  

### Key steps

This repository contains a VQ-VAE model to learn codebook representations of hierarchical structures, and generative attention-diffusion model models to produce microstructural candidates from stress-strain conditioning, and stress-strain results from microstructural input.  This code consists of 3 models

- Model 1 (VQ-VAE to encode hierarchical architected microstructures)
- Model 2 (diffusion model to predict hierarchical architected microstructures from a stress-strain response conditioning)
- Model 3 (diffusion model to predict stress-strain response from a microstructure)

Users should first train the VQ-VAE model (Model 1), then the attention-diffusion models (Models 2 and/or 3). 

##### Reference: 

[1] M. Buehler, A computational building block approach towards multiscale architected materials analysis and design with application to hierarchical metal metamaterials, Modelling and Simulation in Materials Science and Engineering, 2023 

### Overview of the problem solved 

A bioinspired hierarchical honeycomb material is considered in this study, featuring multiple hierarchical levels incorporated into a complex design space. Panel a shows an overview of the hierarchical makeup with four levels of hierarchy ranging from H1…H4.  Panel b summarizes the mechanical boundary condition used to assess mechanical performance by applying compressive loading. By generating a large number of hierarchical designs and associated stress-strain responses, we construct a data set that consists of paired relationships between microstructure images and nonlinear mechanical properties. Panel c summarizes the two problems addressed here. The forward problem produces a stress-strain response based on the input microstructure. In the inverse problem microstructure candidates are generated based on an input, desired, stress-strain response.       

![image](https://user-images.githubusercontent.com/101393859/228824190-d5f5c5f5-babd-4d99-b802-08c4590ddfaa.png)


### How to install and use

```
conda create -n HierarchicalDesign python=3.8
conda activate HierarchicalDesign
```
```
git clone https://github.com/lamm-mit/HierarchicalDesign/
cd HierarchicalDesign
```

Then, install HierarchicalDesign:

```
pip install -e .
```

Start Jupyter Lab (or Jupyter Notebook):

```
jupyter-lab --no-browser
```
Then open the sample Jupyter file and train and/or load pretrained models. 

- Model 1: VQ-VAE model to learn codebook representations of hierarchical structures: VQ_VAE_Microstructure.ipynb
- Model 2: Generative attention-diffusion model: HierarchicalDesignDIffusion_GetMicrostructure.ipynb
- Model 3: Attention-diffusion model to predict stress-strain response from microstructure: HierarchicalDesignDIffusion_GetStressStrain.ipynb 

### Details on the architecture and approach

The figure shows an overview of the neural network architecture used to solve this problem. The model consists of two parts. First (panel a), a vector quantized variational autoencoder (VQ-VAE) architecture that learns to encode microstructure images into a lower-dimensional latent space. We use a discrete approach here that encodes data into a discrete codebook representation that consists of a one-dimensional vector of length N where each entry is one of n_c possible “words” in the design language that defines the microstructures.  The encoder and decoder blocks each consist of a deep neural network featuring convolutional and attention layers. The VQ-VAE model is trained based on unlabeled data of microstructure images. In the next step (panel b), the pre-trained VQ-VAE model is used as an encoding mechanism to train a diffusion model, where it learns how to produce codebook representations that satisfy a certain conditioning. During training, pairs of conditioning and codebook representations of microstructures are used to minimize the reconstruction loss. Once trained (panel c), the model can be used to generate microstructure solutions based on a certain conditioning stress-strain laws. The stress-strain response is encoded as a series of normalized floating point numbers, concatenated with Fourier positional encoding. An identical model is developed and trained also for the forward problem, where the conditioning is the input microstructure, and the diffusion model produces stress-strain responses. 

![image](https://user-images.githubusercontent.com/101393859/228824011-86f1e866-5cce-4b90-9c9e-64ed88fcab68.png)


### Datasets, weights, and additional information

- [Dataset CSV file](https://www.dropbox.com/s/tg4j25rmga4agu4/shc_09_raw_wimagename_5_thick.csv?dl=0) 
- [Microstructure data](https://www.dropbox.com/s/dz1hnhedjocacqu/gene_output_thick.zip?dl=0) 
- [Model 1 weights](https://www.dropbox.com/s/cluk4e4q95laqu2/model-epoch_128.pt?dl=0)
- [Model 2 weights](https://www.dropbox.com/s/nnus6m2z5jbbshv/statedict_save-model-epoch_4000_FINAL.pt?dl=0)
- [Model 3 weights](https://www.dropbox.com/s/u4mojfwp2uxjqkh/statedict_save-model-epoch_610_FINAL.pt?dl=0)

### Overview of the data and format

You need two files to train the model. First, a CSV file that includes both, references to the microstructure data (column "mcirostructure") and stress data (column "stresses"). There is no separate strain data stored; the list of strains per stress increment is identical for all samples and defined in the code. 

Example, a list of stress values: 
```
[2.8133392e-04, 5.0026216e-02, 1.0911082e-01, 1.5260771e-01, 1.9775425e-01,
 2.4043675e-01, 2.8037483e-01, 2.9301879e-01, 2.9600343e-01, 2.9962808e-01,
 3.0461299e-01, 3.0993605e-01, 3.1720343e-01, 3.2225695e-01, 3.2850131e-01,
 3.3622128e-01, 3.4194285e-01, 3.4944272e-01, 3.5820404e-01, 3.6438131e-01,
 3.7371701e-01, 3.8643220e-01, 4.0058151e-01, 4.1546318e-01, 4.3120158e-01,
 4.4801408e-01, 4.6275303e-01, 4.7932705e-01, 4.9467662e-01, 5.1254719e-01,
 5.3236824e-01, 5.5691981e-01]
 ```
Second, the microstructure data consists of images, each of which is associated to a list of stress data as defined in the CSV file. 

### Acknowledgements 

This code is based on [https://github.com/lucidrains/imagen-pytorch](https://github.com/lucidrains/imagen-pytorch) and [https://github.com/lamm-mit/DynaGen](https://github.com/lamm-mit/DynaGen). 

```
@article{BuehlerMSMSE_2023,
    title   = {A computational building block approach towards multiscale architected materials analysis and design with application to hierarchical metal metamaterials},
    author  = {M.J. Buehler},
    journal = {Modelling and Simulation in Materials Science and Engineering},
    year    = {2023},
    volume  = {},
    pages   = {},
    url     = {}
}
```
