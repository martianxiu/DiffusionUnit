# Interpretable Edge Enhancement and Suppression Learning for 3D Point Cloud Segmentation
Created by: Haoyi Xiu, Xin Liu, Weimin Wang, Kyoung-Sook Kim, Takayuki Shinohara, Qiong Chang, and Masashi Matsuoka

<!-- ![concept](figures/concept.jpeg) -->
<p align='center'>
<img src="figures/concept.jpeg" alt="concept" width="300"/>

## Introduction
This repository contains the implementation of [our paper](https://arxiv.org/abs/2209.09483).

## Installation
The code has been tested with Ubuntu 16.04.6, python 3.7, CUDA 11.1.1

We use anaconda to create a virtual environment. 
```
conda create -n diffusion_unit python=3.7
conda activate diffusion_unit
```

Install libraries:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx -y
pip install numpy --upgrade
```

Following [Point Transformer repo](https://github.com/POSTECH-CVLab/point-transformer) to install pointops: 

```
cd lib/pointops
python3 setup.py install
```
## Scene segmentation
Download [dataset](https://drive.google.com/uc?export=download&id=1KUxWagmEWnvMhEb4FRwq2Mj0aa3U3xUf) and put the uncompressed data to DiffusionUnit/s3dis/dataset/s3dis/

### Training
sh start_training.sh

### Testing
sh predict.sh

### Pretrained model and test log
The pretrained model and test log of scene segmentation are available [here](https://staff.aist.go.jp/xin.liu/files/DU/s3dis.zip). 

## Citation
If you find our work useful, please consider citing:
```
@article{xiu2022interpretable,
  title={Interpretable Edge Enhancement and Suppression Learning for 3D Point Cloud Segmentation},
  author={Xiu, Haoyi and Liu, Xin and Wang, Weimin and Kim, Kyoung-Sook and Shinohara, Takayuki and Chang, Qiong and Matsuoka, Masashi},
  journal={arXiv preprint arXiv:2209.09483},
  year={2022}
}
```
## License
Our code is released under MIT License (see LICENSE file for details).

## Acknowledgment
This repo is based on&inspired by the great works of [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer), [KPConv](https://github.com/POSTECH-CVLab/point-transformer), and [dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch).
