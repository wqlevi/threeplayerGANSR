# A Three-Play GAN for Super-resolution in Magnetics Resonance Imaging
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-3918/)
[![PyTorch](https://img.shields.io/badge/PyTorch-grey.svg?logo=PyTorch)](https://pytorch.org/blog/pytorch-1.9-released/) 
[![Paper](http://img.shields.io/badge/Paper-arxiv.2211.16198-B31B1B.svg)](https://arxiv.org/abs/2303.13900)
[![Proceeding](https://img.shields.io/badge/Proceeding-MICCAI2023-blue)](https://link.springer.com/chapter/10.1007/978-3-031-44858-4_3)

The official repository for the paper "A Three-player GAN for Super-Resolution in Magnetics Resonance Imaging" in MLCN workshop of MICCAI2023

# Usage
0. [Patch the entire brain volume into smaller ones, to be compatible with GPU memories](#Patching);
1. [Training on the patches](#Training)
2. [Inferring the test data](#Inferring)
3. [Assembling testing patches](#Assembling)

## Patching
```python
python crop_nifti_9t.py <your data folder path>
```
or in multiprocessing way:
```python
python mp_crop_nifti.py <your data folder path>
```

## Training
```python
python ./mains/ln_DDP_train.py --model_name 'ThreePlayerGAN'
```
this loads the configure YAML file in `./config` folder, of course you can write your own config file or even the training script.

## Inferring
```python
python ./mains/inference/inference_WholeBrain.py [argvs] # skipping patching and assembling, memory-UNfriendly, but you can trade-off it with speed by placing them on CPU
# or
python ./mains/inference/inference.py [argvs]  # also including pathcing and assembling, but trivial difference between stiched patches exist
```
please refer to [utils README](./mains/utils/README.md) for detailed inference introduction.
## Assembling
It mainly serves as an utility module for the inference steps, mainly stored in `./mains/inference/assemble_einops.py`
# Citation

~~~bibtex
@InProceedings{threeplayergan,
  author="Wang, Qi and Mahler, Lucas and Steiglechner, Julius and Birk, Florian and Scheffler, Klaus and Lohmann, Gabriele",
  title="A Three-Player GAN for Super-Resolution in Magnetic Resonance Imaging",
  booktitle="Machine Learning in Clinical Neuroimaging",
  year="2023",
  publisher="Springer Nature Switzerland",
  pages="23--33",
  isbn="978-3-031-44858-4"
}
~~~
