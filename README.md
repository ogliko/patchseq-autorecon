### [Automated reconstruction of dendritic and axonal arbors reveals molecular correlates of neuroanatomy](https://www.biorxiv.org/content/10.1101/2022.03.07.482900v1)
Olga Gliko, Matt Mallory, Rachel Dalley, Rohan Gala, James Gornet, Hongkui Zeng, Staci Sorensen, Uygar Sümbül

#### Abstract
Neuronal anatomy is central to the organization and function of brain cell types. However, anatomical variability within apparently homogeneous populations of cells can obscure such insights. Here, we report large-scale automation of arbor reconstruction on a dataset of 802 inhibitory neurons characterized using the Patch-seq method, which enables measurement of multiple properties from individual neurons, including local morphology and transcriptional signature. We demonstrate that these automated reconstructions can be used in the same manner as manual reconstructions to understand the relationship between many cellular properties used to define cell types. We uncover molecular correlates of laminar innervation on multiple molecularly defined neuronal subclasses and types. In particular, our results reveal molecular correlates of the variability in Layer 1 (L1) innervation even in a transcriptomically defined subpopulation of Martinotti cells in the adult mouse neocortex.

#### About this repository
This repository contains codes and processed data files for analyses presented in the bioRxiv publication.

### Data

 - [Neurons in Mouse Primary Visual Cortex](https://portal.brain-map.org/explore/classes/multimodal-characterization)
 - `data` contains dataset of automated morphological reconstructions of inhibitory neurons
 
 
### Code

#### Neuron reconstruction

**Volumetric data generation**

Matlab functions and scripts to generate volumetric labels from manual traces using a topology preserving fast marching algorithm.
Original github repo [here](https://github.com/rhngla/topo-preserve-fastmarching).

**Segmentation**

Code for training a neuron network model to perform a multi-class segmentation of neuronal arbors is in the pytorch_segment section of this repository.
Original github repo [here](https://github.com/jgornet/NeuroTorch).

**Postprocessing**

Code for postprocessing including relabeling to improve axon/dendrite node assignment of the initial reconstruction is in the postprocessing section of this repository.

**Neuron reconstruction pipeline**

Automated pipeline combines pre-processing raw images, segmentation of raw image stack into soma/axon/dendrite channels, post-processing, and conversion of images to swc file. This code, and [a small example](https://github.com/ogliko/patchseq-autorecon/blob/master/pipeline/example_pipeline.sh) can be found under the pipeline section of this repository. The example's maximum intensity projection (mip) is seen [here](https://github.com/ogliko/patchseq-autorecon/blob/master/pipeline/Example_Specimen_2112/example_specimen.PNG) 
 
#### Data analysis

 - arbor density representation
 - supervised classification
 - sparse feature selection