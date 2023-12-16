# DETR:End-to-End-Object-Detection-with-Transformers

DETR(DETection TRANSformer)을 위한 PyTorch 훈련 모델입니다. 전체 복잡한 수작업 객체 탐지 파이프라인을 Transformer로 교체하고 Faster R-CNN을 ResNet-50과 일치시켜 절반의 계산 능력(FLOP)과 동일한 수의 매개변수를 사용하여 COCO에서 42개의 AP를 얻습니다. PyTorch의 50개 라인에서 추론합니다.

![image](https://github.com/hyeyun0302/DETR_End-to-End-Object-Detection-with-Transformers/assets/104217871/d0e32cc8-52f6-42a4-a13d-43e73d608e2b)


기존의 컴퓨터 비전 기술과 달리 DETR은 객체 탐지를 직접 set prediction 문제로 접근합니다. 이는 초당적 매칭을 통해 고유한 예측을 강제하는 집합 기반 전역 손실과 트랜스포머 인코더-디코더 아키텍처로 구성됩니다. 학습된 객체 쿼리의 고정된 작은 집합이 주어지면 객체의 관계와 글로벌 이미지 컨텍스트가 최종 예측 집합을 직접 병렬로 출력해야 하는 이유가 됩니다. 이러한 병렬 특성으로 인해 DETR은 매우 빠르고 효율적입니다.

객체 검출이 분류보다 어렵지 않고, 훈련과 추론을 위해 복잡한 라이브러리를 필요로 해서는 안 된다고 생각합니다. DETR은 구현과 실험이 매우 간단하며, 저희는 DETR로 추론하는 방법을 PyTorch 코드의 몇 가지 라인에서만 보여주는 독립형 Colab Notebook을 제공합니다. 훈련 코드는 라이브러리가 아니라 표준 훈련 루프를 가진 main.py 가져오기 모델 및 기준 정의라는 아이디어를 따릅니다.

추가적으로 d2/ 폴더에 Detectron2 래퍼를 제공합니다.


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
