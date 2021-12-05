# YOLOF && SSD Implementation in Pytorch

This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the projects [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) , [NVIDIA Apex](https://github.com/NVIDIA/apex) and [YOLOF](https://github.com/chensnathan/YOLOF).
The design goal is modularity and extensibility.

Currently, it has MobileNetV2, EfficientNet, MobileDet_gpu and VGG based SSD/SSD-Lite implementations. 




## Dependencies
1. Python 3.6+
2. OpenCV
3. Pytorch 1.0 or Pytorch 0.4+
4. Caffe2
5. Pandas
6. Boto3 if you want to train models on the Google OpenImages Dataset.
7. [Apex](https://github.com/NVIDIA/apex)
8. [Jstat](https://github.com/rbonghi/jetson_stats)
9. [torchstat](https://github.com/Swall0w/torchstat)
10. [Mish-cuda](https://github.com/thomasbrandon/mish-cuda)
11. [jetson_stats](https://github.com/rbonghi/jetson_stats)



## How to Install

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```



## ![Support-Matrix-NVIDIA-Deep-Learning-TensorRT-Documentation](/home/oem/Pictures/Support-Matrix-NVIDIA-Deep-Learning-TensorRT-Documentation.png)

## Run the demo

### Run the live MobileNetV2 SSD Lite demo

```bash
wget -P models https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
python run_ssd_live_demo.py mb2-ssd-lite models/mb2-ssd-lite-mp-0_686.pth models/voc-model-labels.txt 
```

The above MobileNetV2 SSD-Lite model is not ONNX-Compatible, as it uses Relu6 which is not supported by ONNX.
The code supports the ONNX-Compatible version. Once I have trained a good enough MobileNetV2 model with Relu, I will upload
the corresponding Pytorch and Caffe2 models.

You may notice MobileNetV2 SSD/SSD-Lite is slower than MobileNetV1 SSD/Lite on PC. However, MobileNetV2 is faster on mobile devices.

### Run the live MobileNetV2 YOLOF Lite demo

```
python run_live_demo.py ef-yolof  path/to/your/model.pth  models/voc-model-labels.txt path/to/your/video.mp4
```



## Pretrained Models

### 

| backbone                 | mAP(%) | Param(M) | MAdds(G) |  FLOPS  | Size(MB) |
| ------------------------ | :----: | :------: | :------: | :-----: | :------: |
| mobilenet2-SSDlite       |  55.8  |   3.09   |   1.3    | 663.42M |  144.21  |
| mobilenet2-YOLOF         |        |   8.07   |   2.34   |  1.19G  |  150.38  |
| mobilenet3-small-SSDlite |        |          |          |         |          |
| mobilenet3-small-YOLOF   |        |          |          |         |          |
| mobilenet3-large-SSDlite |  54.3  |   2.16   |   1.04   | 525.5M  |  86.34   |
| mobilenet3-large-YOLOF   |        |   5.98   |   1.33   | 666.67M |  42.58   |
| efficientnet-SSDlite-B0  |  51.7  |   4.84   |   1.6    | 804.8M  |  143.32  |
| efficientnet-YOLOF-B0    |        |   9.86   |   2.66   |  1.34G  |  153.58  |
| mobiledet-SSDlite        |        |   3.07   |   2.22   |  1,12G  |  89.72   |
| mobiledet-YOLOF          |        |   8.1    |   3.28   |  1.65G  |  102.35  |



### 

The code to re-produce the model:

```bash
python train_ssd.py --dataset_type voc  --datasets ~/data/VOC0712/VOC2007 ~/data/VOC0712/VOC2012 --validation_dataset ~/data/VOC0712/test/VOC2007/ --net mb2-ssd-lite --base_net models/mb2-imagenet-71_8.pth  --scheduler cosine --lr 0.01 --t_max 200 --validation_epochs 5 --num_epochs 300
```



```bash
python train_ssd.py --datasets ~/data/VOCkitti/ --validation_dataset ~/data/VOCkitti/ --net mbd-ssd-lite --base_net /model/to/your/model.pth  --batch_size 32 --num_epochs 300 --scheduler "multi-step” —-milestones “120,160”
```

### 


## Training

```bash
python train_ssd.py --datasets ~/data/VOC0712/VOC2007/ ~/data/VOC0712/VOC2012/ --validation_dataset ~/data/VOC0712/test/VOC2007/ --net mb1-ssd --base_net models/mobilenet_v1_with_relu_69_5.pth  --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200
```
```bash
python train_ssd.py --datasets /home/mnt/CenterNet/data/kitti/training/ --validation_dataset /home/mnt/CenterNet/data/kitti/training/ --net mb1-ssd --base_net models/mobilenet_v1_with_relu_69_5.pth  --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200
```


The dataset path is the parent directory of the folders: Annotations, ImageSets, JPEGImages, SegmentationClass and SegmentationObject. You can use multiple datasets to train.


## Evaluation

```bash
python eval_ssd.py --net mb1-ssd  --dataset ~/data/VOC0712/test/VOC2007/ --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt 
```
'''rqp
python eval_ssd.py --net sq-ssd-lite  --dataset /home/sues/data/kitti/VOCdevkit/VOC2007 --trained_model models/sq-ssd-lite/sq-ssd-lite-Epoch-215-Loss-2.6122971734692975.pth --label_file models/voc-model-labels.txt 
'''

## Convert models to ONNX and Caffe2 models

```bash
python convert_to_caffe2_models.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt 
```

The converted models are models/mobilenet-v1-ssd.onnx, models/mobilenet-v1-ssd_init_net.pb and models/mobilenet-v1-ssd_predict_net.pb. The models in the format of pbtxt are also saved for reference.

## Retrain on Open Images Dataset

Let's we are building a model to detect guns for security purpose.

Before you start you can try the demo.

```bash
wget -P models https://storage.googleapis.com/models-hao/gun_model_2.21.pth
wget -P models https://storage.googleapis.com/models-hao/open-images-model-labels.txt
python run_ssd_example.py mb1-ssd models/gun_model_2.21.pth models/open-images-model-labels.txt ~/Downloads/big.JPG
```


If you manage to get more annotated data, the accuracy could become much higher.

### Download data

```bash
python open_images_downloader.py --root ~/data/open_images --class_names "Handgun,Shotgun" --num_workers 20
```



| Dataset  | Train/Val                                                    | Test                                                         | Annotations                                                  |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| VOC2007  | [Train/Validation Data (439 MB)](http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar) | [Test Data With Annotations (431 MB)](http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar) | [Development Kit](http://pjreddie.com/media/files/VOCdevkit_08-Jun-2007.tar) |
| VOC2012  | [Train/Validation Data (1.9 GB)](http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar) | [Test Data (1.8 GB)](http://pjreddie.com/media/files/VOC2012test.tar) | [Development Kit](http://pjreddie.com/media/files/VOCdevkit_18-May-2011.tar) |
| KITTI    |                                                              |                                                              |                                                              |
| COCO2014 | [2014 Train images [83K/13GB\]](http://images.cocodataset.org/zips/train2014.zip) <br/> [2014 Val images [41K/6GB\]](http://images.cocodataset.org/zips/val2014.zip) | [2014 Test images [41K/6GB\]](http://images.cocodataset.org/zips/test2014.zip) | [2014 Train/Val annotations [241MB\]](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) |
| COCO2017 | [2017 Train images [118K/18GB\]](http://images.cocodataset.org/zips/train2017.zip)<br/>[2017 Val images [5K/1GB\]](http://images.cocodataset.org/zips/val2017.zip) | [2017 Test images [41K/6GB\]](http://images.cocodataset.org/zips/test2017.zip) | [2017 Train/Val annotations [241MB\]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) |



## Retrain on VOC2007

```bash
python train_yolof.py --dataset_type voc --datasets /mnt/Datasets/voc/voc2007/VOCdevkit/VOC2007/ --net mb1-yolof --scheduler cosine --batch_size 24 --lr 0.01 --t_max 200 --num_epochs 200 --base_net_lr 0.001 --validation_epochs 5

python train_ssd.py --datasets ~/data/VOC0712/VOC2007/ ~/data/VOC0712/VOC2012/ --validation_dataset ~/data/VOC0712/test/VOC2007/ --net "mb1-ssd" --base_net models/vgg16_reducedfc.pth  --batch_size 24 --num_epochs 150 --scheduler cosine --lr 0.0012 --t_max 150 --validation_epochs 5
```

You can freeze the base net, or all the layers except the prediction heads. 

```
  --freeze_base_net     Freeze base net layers.
  --freeze_net          Freeze all the layers except the prediction head.
```

You can also use different learning rates 
for the base net, the extra layers and the prediction heads.

```
  --lr LR, --learning-rate LR
  --base_net_lr BASE_NET_LR
                        initial learning rate for base net.
  --extra_layers_lr EXTRA_LAYERS_LR
```

As subsets of open images data can be very unbalanced, it also provides
a handy option to roughly balance the data.

```
  --balance_data        Balance training data by down-sampling more frequent
                        labels.
```

### Test on image

```bash
python run_ssd_example.py mb1-ssd models/mobilenet-v1-ssd-Epoch-99-Loss-2.2184619531035423.pth models/open-images-model-labels.txt ~/Downloads/gun.JPG
```


## ONNX Friendly VGG16 SSD

! The model is not really ONNX-Friendly due the issue mentioned here "https://github.com/qfgaohao/pytorch-ssd/issues/33#issuecomment-467533485"

The Scaled L2 Norm Layer has been replaced with BatchNorm to make the net ONNX compatible.

### Train

The pretrained based is borrowed from https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth .

```bash
python train_ssd.py --datasets ~/data/VOC0712/VOC2007/ ~/data/VOC0712/VOC2012/ --validation_dataset ~/data/VOC0712/test/VOC2007/ --net "vgg16-ssd" --base_net models/vgg16_reducedfc.pth  --batch_size 24 --num_epochs 150 --scheduler cosine --lr 0.0012 --t_max 150 --validation_epochs 5
```

### Eval

```bash
python eval_ssd.py --net vgg16-ssd  --dataset ~/data/VOC0712/test/VOC2007/ --trained_model models/vgg16-ssd-Epoch-115-Loss-2.819455094383535.pth --label_file models/voc-model-labels.txt
```

## TOO

1. NAS
2. AnchorFree.



# Cite

A simple, fast, and efficient object detector **without** FPN.

- The [`cvpods`](https://github.com/Megvii-BaseDetection/cvpods) version can be found in https://github.com/megvii-model/YOLOF.
- The neat and re-organized [`Detectron2`](https://github.com/facebookresearch/Detectron) version of YOLOF is available at [https://github. com/chensnathan/YOLOF](https://github.com/chensnathan/YOLOF).

```tex
@inproceedings{chen2021you,
  title={You Only Look One-level Feature},
  author={Chen, Qiang and Wang, Yingming and Yang, Tong and Zhang, Xiangyu and Cheng, Jian and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}

@article{Xu2021TEYOLOFTA,
  title={TE-YOLOF: Tiny and efficient YOLOF for blood cell detection},
  author={Fanxin Xu and Xiangkui Li and Hang Yang and Yali Wang and Wei Xiang},
  journal={ArXiv},
  year={2021},
  volume={abs/2108.12313}
}
```



