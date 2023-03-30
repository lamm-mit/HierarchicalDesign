# HierarchicalDesign: A computational building block approach towards multiscale architected materials analysis and design with application to hierarchical metal metamaterials    
### Markus J. Buehler
email: mbuehler@mit.edu  

We report a computational approach towards multiscale architected materials analysis and design. A particular challenge in modeling and simulation of materials, and especially the development of hierarchical design approaches, has been to identify ways by which complex multi-level material structures can be effectively modeled. One way to achieve this is to use coarse-graining approaches, where  physical relationships can be effectively described with reduced dimensionality. In this paper we report an integrated deep neural network architecture that first learns coarse-grained representations of complex hierarchical microstructure data via a discrete variational autoencoder and then utilizes an attention-based diffusion model solve both forward and inverse problems, including a capacity to solve degenerate design problems. As an application, we demonstrate the method in the analysis and design of hierarchical highly porous metamaterials within the context of nonlinear stress-strain responses to compressive deformation.  We validate the mechanical behavior and mechanisms of deformation using embedded-atom molecular dynamics simulations carried out for copper and nickel, showing good agreement with the design objectives.  

### Key steps

This repository contains a VQ-VAE model to learn codebook representations of hierarchical structures, and generative attention-diffusion model models to produce microstructural candidates from stress-strain conditioning, and stress-strain results from microstructural input. 

Users should first train the VQ-VAE model, then the attention-diffusion models. 

##### Reference: 

[1] M. Buehler, A computational building block approach towards multiscale architected materials analysis and design with application to hierarchical metal metamaterials, Modelling and Simulation in Materials Science and Engineering, 2023 

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

### Details on the architecture and approach

![image](https://user-images.githubusercontent.com/101393859/228824011-86f1e866-5cce-4b90-9c9e-64ed88fcab68.png)

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
