# DETR:End-to-End-Object-Detection-with-Transformers

- 2023-2 인공지능2 기말과제(정혜윤 2020003945)

- 본 게시물은 DETR 논문을 기반으로, 딥러닝 기초지식이 있는 초보자들이 DETR 모델에 대해 보다 쉽게 이해할 수 있도록 작성한 문서입니다.

- 논문 원본과 예시 코드를 추가 자료로 첨부하였습니다.


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


**1. Object detection set prediction loss**

DETR은 먼저 충분히 큰 N개 종류의 predictions이 이루어진다고 가정합니다. 그러고 나서 예측된 값과 실제값을 이분 매칭하게되고, 이를 최적화하게 됩니다. 이를 위한 수식은 다음과 같습니다.

   
![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/ee77aaae-72d0-4ed0-956d-bfdd7bfd8185)


수식을 살펴보면, y hat 값은 예측값 집합, 그냥 y 값은 실제값 집합입니다. 그래서 매칭 loss 값의 합을 가장 작게 만들어주는 sigma를 찾아주면 되는데, 여기서 sigma는 말 그대로 매칭된 결과로 볼 수 있겠습니다.
매칭된 loss 값은 pair-wise한 matching cost입니다.
결과적으로 sigma를 찾는 task는 최적 할당 문제로 생각할 수 있고, 다시 말해 가중 이분 매칭입니다. 이는 Hungarian algorithm에 따라 이루어집니다.
해당 Task를 수행하면서 결과적으로 생성되는 output은 Class와 Bounding boxes가 나옵니다. 그리고 Bounding box는 일반적으로 중심 좌표, 높이, 너비에 대한 정보를 담습니다. 본 논문에서는 Bounding box 관련 값들을 0과 1 사이로 normalize 합니다.

매칭된 loss 값의 수식도 존재하는데, 이는 다음과 같습니다.


![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/f14f88fd-c1f8-4938-8130-1cd433f25bc7)


이는 class가 잘 맞고, bounding box 또한 잘 맞혔다면 Loss 값이 줄어드는 형태를 띕니다.

Bounding box에 대한 Loss 수식은 다음과 같습니다.


![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/1aaa24c5-8946-4355-aa6c-cabf8678c5fd)


전자, 즉 iou에 대한 Loss는 크기와 상관없이 bounding box를 많이 겹칠 수 있도록 해줍니다. 그리고 L1에 대한 Loss 값 또한 두 개의 bounding box가 유사해지도록 하는 역할을 수행합니다. 대신 크기에 영향을 받곤 합니다. 위에서 살펴본 이분 매칭을 통해 기존 region proposal 혹은 anchor와 같은 heuristic한 방법론과도 잘 맞는다는 것을 알 수 있습니다. 이분 매칭의 결과로 set prediction의 중복된 결과를 피할 수 있게 됩니다.

앞서 말한 내용을 종합한 전체 Hungarian Loss는 다음과 같이 정리가 가능합니다.


![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/5d3493be-af6f-4592-9380-0b0b47e29aa4)



**2. DETR architecture**

![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/db04ce4c-a7a2-4c7a-bf26-04a95e76ddff)

DETR은 ***1) CNN backbone, 2) Transformer encoder, 3) Transformer decoder, 4) FFN(Feed Forward Network)*** 로 구성되어 있습니다. 

***1) CNN backbone***

   먼저 입력이미지 ![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/ac728a3f-3d18-4db3-ab6d-d994a2da5330)를 CNN backbone network에 입력하여, feature map ![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/1a2d1148-4f75-421a-a0a3-21fc8b0f2ae6)를 생성합니다. 이 때 C=2048이며, ![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/8722aad0-ea09-4fb8-a09d-35526dd13ecf) 입니다. 

  
***2) Transformer encoder***

이후 1x1 convolution 연산을 적용하여, C차원의 feature map을 d차원으로 감소시켜 새로운 feature map![image](https://github.com/HY-AI2-Projects/hyeyun/assets/104217871/e37fc70c-9a4d-471d-8092-a381e1673d19) 을 생성합니다. Transformer encoder는 sequence를 입력으로 받기 때문에 z0의 spatial dimension을 collapse(=flatten)하여 크기를 d×HW로 변경시켜줍니다. 각 encoder layer는 multi-head self-atttention module과 feed forward network(FFN)으로 구성되어 있습니다. Transformer 구조는 입력 embedding의 순서와 상관없이 같은 출력값을 생성하는 permutation-invariant 속성이 있기 때문에 encoder layer 입력 전에 입력 embedding에 positional encoding을 더해줍니다. 

   
***3) Transformer decoder***

Transformer decoder는 masking을 통해 다음 token을 예측하는 autoregressive 방법을 사용하는 반면, DETR의 decoder는 N개의 object에 대한 정보를 한번에 출력합니다. Decoder 역시 permutation-invariant하기 때문에 입력으로 받는 embedding으로 object queries라고 불리는 learnt positional encoding을 사용합니다.
object query는 object query feature과 object query positional embedding으로 구성되어 있습니다. object query feature는 decoder에 initial input으로 사용되어, decoder layer를 거치면서 학습됩니다. query positional embedding은 decoder layer에서 attention 연산 시 모든 query feature에 더해집니다. query feature는 학습 시작 시 0으로 초기화(zero-initialized)되며, query positional embedding은 학습 가능(learnable)합니다. 이러한 object queries는 길이가 N으로, decoder에 의해 output embedding으로 변환(transform)되며 이후 FFN을 통해 각각 독립적으로(independently) box coordinate와 class label로 decode됩니다. 이는 각각의 object query는 하나의 객체를 예측하는 region proposal에 대응된다고 볼 수  있습니다. 즉, object queries는 N개의 객체를 예측하기 위한 일종의 prior knowledge로도 볼 수 있습니다.
encoder와 유사하게 object query를 각 attention layer의 입력에 더해줍니다. 이 때 embedding은 self-attention과 encoder-decoder attention을 통해 이미지 내 전체 context에 대한 정보를 사용합니다. 이를 통해 객체 사이의 pair-wise relation을 포착하여 객체간의 전역적(global)인 정보를 모델링하는 것이 가능해집니다. 

   
***4) FFN(Feed Forward Network)***

Decoder에서 출력한 output embedding을 3개의 linear layer와 ReLU activation function으로 구성된 FFN에 입력하여 최종 예측을 수행합니다. FFN은 이미지에 대한 class label과 bounding box에 좌표(normalized center coordinate, width, height)를 예측합니다. 이 때 예측하는 class label 중 ∅은 객체가 포착되지 않은 경우로, "background" class를 의미합니다.

추가적으로 본 논문에서는 Auxiliary decoding losses를 사용해서 성능을 높였다고 합니다.


## 🏷️ Model Zoo


저자들은 우선 Object Detection baseline으로 DETR과 DETR-DC5 모델을 제공합니다. 성능(AP) 은 COCO 2017 val5k을 사용해 평가했으며, 실행 시간(Inference Time) 은 첫 100개의 이미지에 대해 측정됩니다.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50-dc5_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101-dc5_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>


이 모델은 torch hub를 통해서 사용할 수 있습니다. 

```python
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
```

또한, Panoptic segmentation 모델도 제공합니다.


<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>box AP</th>
      <th>segm AP</th>
      <th>PQ</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>38.8</td>
      <td>31.1</td>
      <td>43.4</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth">download</a></td>
      <td>165Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>40.2</td>
      <td>31.9</td>
      <td>44.6</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth">download</a></td>
      <td>165Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>40.1</td>
      <td>33</td>
      <td>45.1</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth">download</a></td>
      <td>237Mb</td>
    </tr>
  </tbody>
</table>


## 🏷️ Notebooks

저자들은 DETR에 대한 이해를 돕기 위해 colab에서 몇 가지의 notebook을 제공합니다.

* [DETR's hands on Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb):
  
1. hub에서 모델을 불러오는 방법
2. 예측을 생성하는 방법
3. 모델의 attention을 시각화하는 방법(논문의 figure와 유사)


* [Standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb):
  
1. 가장 간단한 버전의 DETR을 50 lines of python code로 실행하는 방법
2. 예측을 시각화하는 방법


* [Panoptic Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb):
  
1. Panoptic segmentation을 위한 DETR을 사용하는 방법
2. 예측을 시각화하는 방법


## 🏷️ Usage - Object detection

DETR은 위에서 기술했던 대로 기존의 패키지들에 크게 의존적이지 않습니다. 전반적인 설치 파이프라인은 아래와 같습니다.

1. Repository clone:
```
git clone https://github.com/facebookresearch/detr.git
```
2. install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
3. Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
pycocotolls는 COCO dataset에 evaluation을 하기 위한 툴입니다.

(optional) install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## 🏷️ Data preparation

본 연구에서 대표적으로 사용한 dataset은 COCO 2017입니다. 주석(annotation)이 포함된 train/val image는
[http://cocodataset.org](http://cocodataset.org/#download)에서 다운받을 수 있습니다. 해당 dataset의 structure는 아래와 같아야 합니다.

```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## 🏷️ Training
예시로 node 당 8 gpus를 사용해 300 epoch을 학습시킬 경우 아래와 같은 명령어를 사용하면 됩니다.:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
1 epoch은 28분 정도 걸리기에, 300 epoch은 6일정도 걸릴 수 있습니다(V100 기준).
결과 재생산을 용이하기 하기 위해 저자들은 150 epoch schedule에 대한 results and training logs
[results and training logs](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f)를 제공합니다(39.5/60.3 AP/AP50).

저자들은 transformer를 학습하는 데 1e-4의 학습률을, backbone을 학습하는데 1e-5의 학습률을 적용한 AdamW을 DETR 학습에 적용합니다. Augmentaiton을 위해 Horizontal flips, scales, crops가 쓰였습니다. 이미지들은 최소 800, 최대 1333의 size를 갖게끔 rescaled됩니다. Transformer는 0.1의 dropout을, 전체 모델은 0.1의 grad clip을 사용해 학습됩니다.


## 🏷️ Evaluation
DETR R50을 COCO val5k에 대해 평가하고 싶으면 아래와 같은 명령어를 실행하면 됩니다.:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```
모든 DETR detection model에 대한 평가 결과는 제공합니다.[gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).
단, GPU 당 batch size(number of images)에 따라 결과가 상당히 변합니다. 예를 들어, batch size 1로 학습한 DC5 모델의 경우 GPU 당 1개 이상의 이미지를 사용해 평가할 경우 성능이 굉장히 낮게 나옵니다.

## 🏷️ Reference 
- https://github.com/facebookresearch
