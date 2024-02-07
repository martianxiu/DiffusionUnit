# Diffusion Unit: Interpretable Edge Enhancement and Suppression Learning for 3D Point Cloud Segmentation
Created by: Haoyi Xiu, Xin Liu, Weimin Wang, Kyoung-Sook Kim, Takayuki Shinohara, Qiong Chang, and Masashi Matsuoka

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/interpretable-edge-enhancement-and/3d-part-segmentation-on-shapenet-part)](https://paperswithcode.com/sota/3d-part-segmentation-on-shapenet-part?p=interpretable-edge-enhancement-and)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/interpretable-edge-enhancement-and/semantic-segmentation-on-s3dis-area5)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis-area5?p=interpretable-edge-enhancement-and)

<!-- ![concept](figures/concept.jpeg) -->
<p align='center'>
<img src="figures/concept.jpeg" alt="concept" width="300"/>

## Abstract
3D point clouds are discrete samples of continuous surfaces which can be used for various applications. However, the lack of true connectivity information, i.e., edge information, makes point cloud recognition challenging. Recent edge-aware methods incorporate edge modeling into network designs to better describe local structures. Although these methods show that incorporating edge information is beneficial, how edge information helps remains unclear, making it difficult for users to analyze its usefulness. To shed light on this issue, in this study, we propose a new algorithm called Diffusion Unit (DU) that handles edge information in a principled and interpretable manner while providing decent improvement. First, we theoretically show that DU learns to perform task-beneficial edge enhancement and suppression. Second, we experimentally observe and verify the edge enhancement and suppression behavior. Third, we empirically demonstrate that this behavior contributes to performance improvement. Extensive experiments and analyses performed on challenging benchmarks verify the effectiveness of DU. Specifically, our method achieves state-of-the-art performance in object part segmentation using ShapeNet part and scene segmentation using S3DIS. 

## Paper
You can download our paper from [arXiv](https://arxiv.org/abs/2209.09483) or [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231223009037). 

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
conda install -c anaconda h5py pyyaml -y
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
@article{xiu2023diffusion,
  title={Diffusion unit: Interpretable edge enhancement and suppression learning for 3D point cloud segmentation},
  author={Xiu, Haoyi and Liu, Xin and Wang, Weimin and Kim, Kyoung-Sook and Shinohara, Takayuki and Chang, Qiong and Matsuoka, Masashi},
  journal={Neurocomputing},
  pages={126780},
  year={2023},
  publisher={Elsevier}
}
```
## License
Our code is released under MIT License (see LICENSE file for details).

## Acknowledgment
This repo is based on&inspired by the great works of [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer), [KPConv](https://github.com/POSTECH-CVLab/point-transformer), and [dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch).
