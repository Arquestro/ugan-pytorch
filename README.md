# Underwater Color Correction made with PyTorch(UGAN)

This repository contains PyTorch implementation of UGAN based on pix2pix with WGAN-GP, L1 and IGDL losses.
Main idea and implementation with Tensorflow as an example have been taken from [this repository](https://github.com/cameronfabbri/Underwater-Color-Correction). 

For more materials and experimental results visit [project page](http://irvlab.cs.umn.edu/enhancing-underwater-imagery-using-generative-adversarial-networks) and [arxiv](https://arxiv.org/pdf/1801.04011.pdf).


This repository is in a state of heavy WIP, but it's working on images with 256x256 resolution. No rescaling yet!

## Getting Started

### Prerequisites

Python 3.6 version or higher. Installing modules from requirements should be enough.
```
pip install -r requirements.txt
```

## Running

Parameters are hardcoded for now, feel free to change scripts parameters for your needs. I will make simple configs for experiments in near future. Make sure to have the same batch size on inference as in training or it will be messed up with Batch Normalization(it's an issue, fix will be soon).

### Training

To start training, run the following script:
```
python train.py
```
Make sure that you have loaded the dataset for training and paths to folders with images from distorted and undistorted domain do exist.
Generated dataset, that is used by UGAN, has been reformatted in the way to make training and inference simple and clean. It can be downloaded from [here](https://drive.google.com/open?id=1I_cNABVUXpTx_fVo-s4dybaGLsf9E_PT).

Output should be done trace of TorchScript that can be loaded and used easily with inference pipeline.

### Inference

To start inference, run the following script:
```
python inference.py
``` 
Make sure that output folder exists(will automate later).
!!!WARNING!!!
Batch Normalization in Generator Net seems breaking the inference stage due to not being fixed after tracing with population statistics.

## License

This repository is under [MIT License]().

## Acknowledgments

### Creators of UGAN

* Cameron Fabbri
* Md Jahidul Islam 
* Junaed Sattar

### Ideas and Resources Support

* [MaritimeAI](https://maritimeai.net/) team.
* [Open Data Science](https://ods.ai/) community.
