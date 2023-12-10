# DETR-End-to-End-Object-Detection-with-Transformers
DETR : End-to-End Object Detection with Transformers

#DETR(DETection TRANSformer)을 위한 PyTorch 훈련 코드 및 사전 훈련된 모델입니다. 저희는 전체 복잡한 수작업 객체 탐지 파이프라인을 Transformer로 교체하고 Faster R-CNN을 ResNet-50과 일치시켜 절반의 계산 능력(FLOP)과 동일한 수의 매개변수를 사용하여 COCO에서 42개의 AP를 얻습니다. PyTorch의 50개 라인에서 추론합니다.

![image](https://github.com/hyeyun0302/DETR_End-to-End-Object-Detection-with-Transformers/assets/104217871/d0e32cc8-52f6-42a4-a13d-43e73d608e2b)


What it is. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

About the code. We believe that object detection should not be more difficult than classification, and should not require complex libraries for training and inference. DETR is very simple to implement and experiment with, and we provide a standalone Colab Notebook showing how to do inference with DETR in only a few lines of PyTorch code. Training code follows this idea - it is not a library, but simply a main.py importing model and criterion definitions with standard training loops.

Additionnally, we provide a Detectron2 wrapper in the d2/ folder. See the readme there for more information.

For details see End-to-End Object Detection with Transformers by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.

See our blog post to learn more about end to end object detection with transformers.
