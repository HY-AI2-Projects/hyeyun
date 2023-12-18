# DETR:End-to-End-Object-Detection-with-Transformers

● 2023-2 인공지능2 기말과제(정혜윤 2020003945)

● 본 게시물은 DETR 논문을 기반으로, 딥러닝 기초지식이 있는 초보자들이 DETR 모델에 대해 보다 쉽게 이해할 수 있도록 작성한 문서입니다.

● 논문 원본과 예시 코드를 추가 자료로 첨부하였습니다.


## 🏷️ Introduction

DETR(End-to-End Object Detection with Transformers)은 Facebook Research 팀이 2020년 8월에 컴퓨터 비전 학회인 ECCV에서 발표한 논문입니다. DETR은 Transformer 구조를 활용하여, end-to-end로 object detection을 수행하면서도 높은 성능을 보였습니다. 기존 object detection 방법론은 prior knowledge가 많이 요구되었고, NMS(Non Maximum Suppression)과 같은 post-processing 과정이 반드시 필요했습니다. 반면 DETR은 object detection을 직접 set prediction 문제로 접근합니다. 현재 많은 SOTA 모델들이 DETR을 기반으로 발전한만큼, 반드시 읽어야하는 기념비적인 논문이라고 할 수 있습니다. 



## 🏷️ Contribution 

본 논문에서 주장하는 contribution은 다음과 같습니다.

1. 본 논문에서는 object detection을 direct set prediction으로 정의하여, transformer와 bipartite matching loss(이분매칭)를 사용한 DETR(DEtection TRansformer)을 제안합니다. 
2. DETR은 COCO dataset에 대하여 Faster R-CNN과 비슷한 수준의 성능을 보입니다. 
3. 추가적으로, self-attention을 통한 global information(전역 정보)를 활용함으로써 크기가 큰 객체를 Faster R-CNN보다 훨씬 잘 포착합니다.



## 🏷️ Method 

본 논문에서는 object detection 시 direct set prediction을 위해 두 가지 요소가 필수적이라고 합니다. 

(1) predicted bounding box와 ground truth box 사이의 unique matching을 가능하도록 하는 set prediction loss

(2) 한 번의 forward pass로 object model 사이의 relation을 예측하는 architecture


1. Object detection set prediction loss
   
![image](https://github.com/hyeyun0302/DETR_End-to-End-Object-Detection-with-Transformers/assets/104217871/d0e32cc8-52f6-42a4-a13d-43e73d608e2b)

먼저 첫 번째 조건 (1)을 충족하기 위해 loss를 계산하는 과정은 두 단계로 구분됩니다. 첫 번째로, predicted bounding box와 ground truth box 사이의 unique한 matching을 수행하는 과정입니다. 두 번째 단계에서는 matching된 결과를 기반으로 hungarian loss를 연산합니다. 이 중 먼저 첫 번째 단계부터 살펴보겠습니다. 

1.1 Find optimal matching 

기존의 연구는 수 천개의 anchor를 생성하여, 객체를 예측하기 위한 proposal로 사용하는데 이는 객체가 “얼마나” 있는지 알 수 없기 때문입니다. 본 논문에서 제안한 DETR은 고정된 크기의 
N
개의 prediction만을 수행함으로써, 수많은 anchor를 생성하는 과정을 우회합니다. 이 때 
N
은 일반적으로 이미지 내 존재하는 객체의 수보다 훨씬 더 큰 수로 지정했습니다. 즉, 이는 DETR을 통해 예측하는 객체의 수는 최대 
N
개임을 의미합니다. 이를 통해 적은 수의 prediction이 생성되어, ground truth와의 unique matching을 상대적으로 쉽게 수행할 수 있습니다.




이는 이분 매칭을 통해 고유한 예측을 강제하는 집합 기반 전역 손실과 Transformer encoder-decoder 아키텍처로 구성됩니다. 학습된 객체 쿼리의 고정된 작은 집합이 주어지면 객체의 관계와 글로벌 이미지 컨텍스트가 최종 예측 집합을 직접 병렬로 출력해야 하는 이유가 됩니다. 이러한 병렬 특성으로 인해 DETR은 매우 빠르고 효율적입니다.

객체 검출이 분류보다 어렵지 않고, 훈련과 추론을 위해 복잡한 라이브러리를 필요로 해서는 안 된다고 생각합니다. DETR은 구현과 실험이 매우 간단하며, 저희는 DETR로 추론하는 방법을 PyTorch 코드의 몇 가지 라인에서만 보여주는 독립형 Colab Notebook을 제공합니다. 훈련 코드는 라이브러리가 아니라 표준 훈련 루프를 가진 main.py 가져오기 모델 및 기준 정의라는 아이디어를 따릅니다.


저희는 기본 DETR 및 DETR-DC5 모델을 제공하며, 향후 더 많은 모델을 포함할 계획입니다. AP는 COCO 2017 val5k에서 계산되며, 토치스크립트 변환기를 사용하여 처음 100 val5k COCO 이미지 이상의 추론 시간을 갖습니다.


	name	backbone	schedule	inf_time	box AP	url	size
0	DETR	R50	500	0.036	42.0	model | logs	159Mb
1	DETR-DC5	R50	500	0.083	43.3	model | logs	159Mb
2	DETR	R101	500	0.050	43.5	model | logs	232Mb
3	DETR-DC5	R101	500	0.097	44.9	model | logs	232Mb


COCO val5k 평가 결과는 이 요지에서 확인할 수 있습니다.

이 모델은 토치 허브를 통해서도 사용할 수 있으며, 사전 교육된 중량으로 DERTR50을 로드할 수 있습니다:

model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

COCO panoptic val5k models:

	name	backbone	box AP	segm AP	PQ	url	size
0	DETR	R50	38.8	31.1	43.4	download	165Mb
1	DETR-DC5	R50	40.2	31.9	44.6	download	165Mb
2	DETR	R101	40.1	33	45.1	download	237Mb


##Notebooks

DETR에 대한 이해를 돕기 위해 몇 가지 노트북을 콜라브로 제공합니다:

DETR의 Colab 노트북 활용: 허브에서 모델을 로드하고 예측을 생성한 다음 모델의 주의력을 시각화하는 방법을 보여줍니다(논문의 그림과 유사)
독립형 콜랩 노트북: 이 노트북에서는 DETR의 단순화된 버전을 50줄의 파이썬으로 구현한 다음 예측을 시각화하는 방법을 시연합니다. 코드베이스에 들어가기 전에 아키텍처를 더 잘 이해하고 주변을 돌아다녀야 한다면 좋은 출발점이 될 것입니다.
Panoptic Colab 노트북: Panoptic 분할 및 plo를 위해 DETR을 사용하는 방법을 시연합니다t the predictions.

##Usage - Object detection
DRIG에는 컴파일된 추가 구성 요소가 없고 패키지 종속성이 최소이므로 코드 사용이 매우 간단합니다. 저희는 콘다를 통해 종속성을 설치하는 방법을 안내합니다. 먼저 저장소를 로컬로 복제합니다:

git clone https://github.com/facebookresearch/detr.git
install PyTorch 1.5+ and torchvision 0.6+:

conda install -c pytorch pytorch torchvision
Install pycocotools (for evaluation on COCO) and scipy (for training):
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

detection 모델을 훈련하고 평가합니다.

(optional) to work with panoptic install panopticapi:
pip install git+https://github.com/cocodataset/panopticapi.git


<div align="center">
  <img src="./assets/logo_2.png" width="30%">
</div>
<h2 align="center">🦖detrex: Benchmarking Detection Transformers</h2>
<p align="center">
    <a href="https://github.com/IDEA-Research/detrex/releases">
        <img alt="release" src="https://img.shields.io/github/v/release/IDEA-Research/detrex">
    </a>
    <a href="https://detrex.readthedocs.io/en/latest/index.html">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href='https://detrex.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/detrex/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://github.com/IDEA-Research/detrex/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/IDEA-Research/detrex.svg?color=blue">
    </a>
    <a href="https://github.com/IDEA-Research/detrex/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="https://github.com/IDEA-Research/detrex/issues">
        <img alt="open issues" src="https://img.shields.io/github/issues/IDEA-Research/detrex">
    </a>
</p>


<div align="center">

<!-- <a href="https://arxiv.org/abs/2306.07265">📚Read detrex Benchmarking Paper</a> <sup><i><font size="3" color="#FF0000">New</font></i></sup> |
<a href="https://rentainhe.github.io/projects/detrex/">🏠Project Page</a> <sup><i><font size="3" color="#FF0000">New</font></i></sup> |  [🏷️Cite detrex](#citation) -->

[📚Read detrex Benchmarking Paper](https://arxiv.org/abs/2306.07265) | [🏠Project Page](https://rentainhe.github.io/projects/detrex/) | [🏷️Cite detrex](#citation) | [🚢DeepDataSpace](https://github.com/IDEA-Research/deepdataspace)

</div>


<div align="center">

[📘Documentation](https://detrex.readthedocs.io/en/latest/index.html) |
[🛠️Installation](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html) |
[👀Model Zoo](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) |
[🚀Awesome DETR](https://github.com/IDEA-Research/awesome-detection-transformer) |
[🆕News](#whats-new) |
[🤔Reporting Issues](https://github.com/IDEA-Research/detrex/issues/new/choose)

</div>


## Introduction

detrex is an open-source toolbox that provides state-of-the-art Transformer-based detection algorithms. It is built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and its module design is partially borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection) and [DETR](https://github.com/facebookresearch/detr). Many thanks for their nicely organized code. The main branch works with **Pytorch 1.10+** or higher (we recommend **Pytorch 1.12**).

<div align="center">
  <img src="./assets/detr_arch.png" width="100%"/>
</div>

<details open>
<summary> Major Features </summary>

- **Modular Design.** detrex decomposes the Transformer-based detection framework into various components which help users easily build their own customized models.

- **Strong Baselines.** detrex provides a series of strong baselines for Transformer-based detection models. We have further boosted the model performance from **0.2 AP** to **1.1 AP** through optimizing hyper-parameters among most of the supported algorithms.

- **Easy to Use.** detrex is designed to be **light-weight** and easy for users to use:
  - [LazyConfig System](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more flexible syntax and cleaner config files.
  - Light-weight [training engine](./tools/train_net.py) modified from detectron2 [lazyconfig_train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py)

Apart from detrex, we also released a repo [Awesome Detection Transformer](https://github.com/IDEA-Research/awesome-detection-transformer) to present papers about Transformer for detection and segmentation.

</details>

## Fun Facts
The repo name detrex has several interpretations:
- <font color=blue> <b> detr-ex </b> </font>: We take our hats off to DETR and regard this repo as an extension of Transformer-based detection algorithms.

- <font color=#db7093> <b> det-rex </b> </font>: rex literally means 'king' in Latin. We hope this repo can help advance the state of the art on object detection by providing the best Transformer-based detection algorithms from the research community.

- <font color=#008000> <b> de-t.rex </b> </font>: de means 'the' in Dutch. T.rex, also called Tyrannosaurus Rex, means 'king of the tyrant lizards' and connects to our research work 'DINO', which is short for Dinosaur.

## What's New
v0.5.0 was released on 16/07/2023:
- Support [Focus-DETR (ICCV'2023)](./projects/focus_detr/).
- Support [SQR-DETR (CVPR'2023)](https://github.com/IDEA-Research/detrex/tree/main/projects/sqr_detr), credits to [Fangyi Chen](https://github.com/Fangyi-Chen)
- Support [Align-DETR (ArXiv'2023)](./projects/align_detr/), credits to [Zhi Cai](https://github.com/FelixCaae)
- Support [EVA-01 (CVPR'2023 Highlight)](https://github.com/baaivision/EVA/tree/master/EVA-01) and [EVA-02 (ArXiv'2023)](https://github.com/baaivision/EVA/tree/master/EVA-02) backbones, please check [DINO-EVA](./projects/dino_eva/) for more benchmarking results.

Please see [changelog.md](./changlog.md) for details and release history.

## Installation

Please refer to [Installation Instructions](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html) for the details of installation.

## Getting Started

Please refer to [Getting Started with detrex](https://detrex.readthedocs.io/en/latest/tutorials/Getting_Started.html) for the basic usage of detrex. We also provides other tutorials for:
- [Learn about the config system of detrex](https://detrex.readthedocs.io/en/latest/tutorials/Config_System.html)
- [How to convert the pretrained weights from original detr repo into detrex format](https://detrex.readthedocs.io/en/latest/tutorials/Converters.html)
- [Visualize your training data and testing results on COCO dataset](https://detrex.readthedocs.io/en/latest/tutorials/Tools.html#visualization)
- [Analyze the model under detrex](https://detrex.readthedocs.io/en/latest/tutorials/Tools.html#model-analysis)
- [Download and initialize with the pretrained backbone weights](https://detrex.readthedocs.io/en/latest/tutorials/Using_Pretrained_Backbone.html)
- [Frequently asked questions](https://github.com/IDEA-Research/detrex/issues/109)
- [A simple onnx convert tutorial provided by powermano](https://github.com/IDEA-Research/detrex/issues/192)
- Simple training techniques: [Model-EMA](https://github.com/IDEA-Research/detrex/pull/201), [Mixed Precision Training](https://github.com/IDEA-Research/detrex/pull/198), [Activation Checkpoint](https://github.com/IDEA-Research/detrex/pull/200)
- [Simple tutorial about custom dataset training](https://github.com/IDEA-Research/detrex/pull/187)

Although some of the tutorials are currently presented with relatively simple content, we will constantly improve our documentation to help users achieve a better user experience.

## Documentation

Please see [documentation](https://detrex.readthedocs.io/en/latest/index.html) for full API documentation and tutorials.

## Model Zoo
Results and models are available in [model zoo](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html).

<details open>
<summary> Supported methods </summary>

- [x] [DETR (ECCV'2020)](./projects/detr/)
- [x] [Deformable-DETR (ICLR'2021 Oral)](./projects/deformable_detr/)
- [x] [PnP-DETR (ICCV'2021)](./projects/pnp_detr/)
- [x] [Conditional-DETR (ICCV'2021)](./projects/conditional_detr/)
- [x] [Anchor-DETR (AAAI 2022)](./projects/anchor_detr/)
- [x] [DAB-DETR (ICLR'2022)](./projects/dab_detr/)
- [x] [DAB-Deformable-DETR (ICLR'2022)](./projects/dab_deformable_detr/)
- [x] [DN-DETR (CVPR'2022 Oral)](./projects/dn_detr/)
- [x] [DN-Deformable-DETR (CVPR'2022 Oral)](./projects/dn_deformable_detr/)
- [x] [Group-DETR (ICCV'2023)](./projects/group_detr/)
- [x] [DETA (ArXiv'2022)](./projects/deta/)
- [x] [DINO (ICLR'2023)](./projects/dino/)
- [x] [H-Deformable-DETR (CVPR'2023)](./projects/h_deformable_detr/)
- [x] [MaskDINO (CVPR'2023)](./projects/maskdino/)
- [x] [CO-MOT (ArXiv'2023)](./projects/co_mot/)
- [x] [SQR-DETR (CVPR'2023)](./projects/sqr_detr/)
- [x] [Align-DETR (ArXiv'2023)](./projects/align_detr/)
- [x] [EVA-01 (CVPR'2023 Highlight)](./projects/dino_eva/)
- [x] [EVA-02 (ArXiv'2023)](./projects/dino_eva/)
- [x] [Focus-DETR (ICCV'2023)](./projects/focus_detr/)

Please see [projects](./projects/) for the details about projects that are built based on detrex.

</details>


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
- detrex is an open-source toolbox for Transformer-based detection algorithms created by researchers of **IDEACVR**. We appreciate all contributions to detrex!
- detrex is built based on [Detectron2](https://github.com/facebookresearch/detectron2) and part of its module design is borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection), [DETR](https://github.com/facebookresearch/detr), and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).


## Citation
If you use this toolbox in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

- Citing **detrex**:

```BibTeX
@misc{ren2023detrex,
      title={detrex: Benchmarking Detection Transformers}, 
      author={Tianhe Ren and Shilong Liu and Feng Li and Hao Zhang and Ailing Zeng and Jie Yang and Xingyu Liao and Ding Jia and Hongyang Li and He Cao and Jianan Wang and Zhaoyang Zeng and Xianbiao Qi and Yuhui Yuan and Jianwei Yang and Lei Zhang},
      year={2023},
      eprint={2306.07265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<details>
<summary> Citing Supported Algorithms </summary>

```BibTex
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}

@inproceedings{
  zhu2021deformable,
  title={Deformable {\{}DETR{\}}: Deformable Transformers for End-to-End Object Detection},
  author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=gZ9hCDWe6ke}
}

@inproceedings{meng2021-CondDETR,
  title       = {Conditional DETR for Fast Training Convergence},
  author      = {Meng, Depu and Chen, Xiaokang and Fan, Zejia and Zeng, Gang and Li, Houqiang and Yuan, Yuhui and Sun, Lei and Wang, Jingdong},
  booktitle   = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year        = {2021}
}

@inproceedings{
  liu2022dabdetr,
  title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
  author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}

@inproceedings{li2022dn,
  title={Dn-detr: Accelerate detr training by introducing query denoising},
  author={Li, Feng and Zhang, Hao and Liu, Shilong and Guo, Jian and Ni, Lionel M and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13619--13627},
  year={2022}
}

@inproceedings{
  zhang2023dino,
  title={{DINO}: {DETR} with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel Ni and Heung-Yeung Shum},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=3mRwyG5one}
}

@InProceedings{Chen_2023_ICCV,
  author    = {Chen, Qiang and Chen, Xiaokang and Wang, Jian and Zhang, Shan and Yao, Kun and Feng, Haocheng and Han, Junyu and Ding, Errui and Zeng, Gang and Wang, Jingdong},
  title     = {Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
  pages     = {6633-6642}
}

@InProceedings{Jia_2023_CVPR,
  author    = {Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  title     = {DETRs With Hybrid Matching},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {19702-19712}
}

@InProceedings{Li_2023_CVPR,
  author    = {Li, Feng and Zhang, Hao and Xu, Huaizhe and Liu, Shilong and Zhang, Lei and Ni, Lionel M. and Shum, Heung-Yeung},
  title     = {Mask DINO: Towards a Unified Transformer-Based Framework for Object Detection and Segmentation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {3041-3050}
}

@article{yan2023bridging,
  title={Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking},
  author={Yan, Feng and Luo, Weixin and Zhong, Yujie and Gan, Yiyang and Ma, Lin},
  journal={arXiv preprint arXiv:2305.12724},
  year={2023}
}

@InProceedings{Chen_2023_CVPR,
  author    = {Chen, Fangyi and Zhang, Han and Hu, Kai and Huang, Yu-Kai and Zhu, Chenchen and Savvides, Marios},
  title     = {Enhanced Training of Query-Based Object Detection via Selective Query Recollection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {23756-23765}
}
```


</details>



