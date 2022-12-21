# Yolov7-MindSpore

本项目为yolov7目标检测算法在华为MindSpore框架下的实现。对应[论文](https://arxiv.org/abs/2004.10934)为：\
Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696, 2022.

## 环境配置
这里采用的MindSpore版本为MindSpore1.8.1，硬件平台为GPU CUDA11.1，操作系统为LINUX-x86_64，编程语言Python3.8，使用Conda方式安装环境。MindSpore安装详见[官网](https://www.mindspore.cn/install)。

此处安装MindSpore时创建的conda虚拟环境为*mindspore_py38*，通过运行`conda activate mindspore_py38`进入对应的conda环境，并安装与CUDA版本对应的Pytorch。yolov7所需的其他库可参照官方给出的`requirements.txt`，在终端输入
```
  pip install -r requirements.txt
```
进行安装。安装过后，可以再次输入上述命令检验是否全部安装成功。

## 数据集
这里采用COCO格式的数据集，可以直接从[COCO2017](https://cocodataset.org/#download)下载完整的MS COCO数据集。或者为了更快捷的进行训练，我们可以对COCO数据集进行精简；或者根据需求制作自己的数据集。
### 精简数据集
在项目文件夹下创建`dataset`文件夹，用于存放数据集。将下载好的完整版COCO2017数据集存入其中，然后运行`others`文件夹下的`division.py`划分出小型的COCO数据集。这里我们将其命名为`tiny_coco`。

需要注意的是，用户需根据项目文件夹在个人电脑上的路径对`division.py`中的相关量进行修改。并根据自身需要指定划分的图片数量。

### 自制数据集
这里给出自制数据集的思路，以便供用户根据自身需求制作COCO格式的数据集。

首先在`dataset`文件夹下建立一个文件夹，进入该文件夹并创建`annotations`、`images`、`labels`文件夹。然后分别在`images`、`labels`文件夹下各自再建`train`、`val`文件夹。在`images`文件夹下按照
`9:1`的数量比例分别将训练图片存入`train`和`val`文件夹。接着制作标签并存入`labels`下的`train`和`val`文件夹。制作标签可用`labelImg`工具，安装方式为：
```
  pip install labelImg
```
这里将选择标签格式为VOC（因为labelImg不支持直接生成COCO格式）。最后将VOC格式转换为COCO格式，生成对应的`.json`文件存入`annotations`文件夹下。

### 数据集目录
这里采用自行精简后的小型COCO数据集`tiny_coco`，数据集目录如下。
```
      ├── dataset
        ├── tiny_coco
            ├── annotations
            │   ├─ tiny_instances_train.json
            │   └─ tiny_instances_val.json
            ├── images
            │   ├─train
            │   │   ├─picture1.jpg
            │   │   ├─ ...
            │   │   └─picturen.jpg
            │   └─ val
            │       ├─picture1.jpg
            │       ├─ ...
            │       └─picturen.jpg
            ├── labels
            │   ├─train
            │   │   ├─label1.txt
            │   │   ├─ ...
            │   │   └─labeln.txt
            │   └─ val
            │       ├─label1.txt
            │       ├─ ...
            │       └─labeln.txt
            ├── train_tiny.txt
            └── val_tiny.txt
```

### 其他
制作好数据集后，需要在`config/data`和`config/network_yolov7`下分别新增`coco_tiny.yaml`和`yolov7_tiny.yaml`。可直接复制原先的`coco.yaml`和`yolov7.yaml`。对于`coco_tiny.yaml`，修改`train_tiny.txt`和`val_tiny.txt`所在文件路径，根据标签种类设置`nc`和`class name`。对于对于`yolov7_tiny.yaml`，修改`nc`。


## 训练
训练相关的参数可以通过`config`下的`args.py`设置。这里`device_target`采用`GPU`，如果修改为其他设备，不仅需要修改`args.py`里的`--device_target`，还需要对代码中出现的诸如`opt.device_target == "GPU"`进行修改。相关设置如下：
```
--device_target = GPU
--cfg = ./config/network_yolov7/yolov7_tiny.yaml
--data = ./config/data/coco_tiny.yaml
-hyp = ./config/data/hyp.scratch.p5.yaml
--recompute = True
--recompute_layers = 5
--batch_size = 1
--epochs = 200
```
采用pycharm进行训练。完成设置后，运行`train.py`进行训练。或在终端输入
```
python train.py > log.txt 2>&1
```
进行训练并导出过程日志。这里将一次训练日志放在`others`里。训练中的输出情况例如：
```
Epoch 200/1, Step 231/1, size (640, 640), fp/bp time cost: 720.80 ms
Epoch 200/1, Step 231/1, size (640, 640), loss: 27.2907, lbox: 0.0849, lobj: 2.7001, lcls: 0.6263, cur_lr: [0.00000000, 0.00000000, 0.10000000], step time: 721.23 ms
Epoch 200/1, Step 231/2, size (640, 640), fp/bp time cost: 470.73 ms
Epoch 200/1, Step 231/2, size (640, 640), loss: 27.2405, lbox: 0.0797, lobj: 2.6989, lcls: 0.6264, cur_lr: [0.00000216, 0.00000216, 0.09990216], step time: 476.26 ms
Epoch 200/1, Step 231/3, size (640, 640), fp/bp time cost: 426.39 ms
Epoch 200/1, Step 231/3, size (640, 640), loss: 27.2644, lbox: 0.0864, lobj: 2.6964, lcls: 0.6253, cur_lr: [0.00000433, 0.00000433, 0.09980433], step time: 430.63 ms
```
这里在个人电脑上进行训练，总耗时约5小时。

## 测试
这里给出基于`tiny_coco`数据集训练最后得到的模型权重文件`yolov7_tiny_ours.ckpt`。运行`test.py`分别得到混淆矩阵confusion_matrix、查准率和召回率的调和平均数F1-score、准确率和置信度的关系图P_curve、精准率与召回率的关系图PR_curve以及召回率和置信度之间的关系图R_curve。

这里对`args.py`中有关`test`的主要参数设置如下：
```
--weights = yolov7_tiny_ours.ckpt
--cfg = ./config/network_yolov7/yolov7_tiny.yaml
--data = ./config/data/coco_tiny.yaml
--task = val
```
也可根据自身需求对上述参数进行修改，例如在终端运行
```
python test.py --weights = your_model_name.ckpt --cfg = ./config/network_yolov7/file.yaml --data = ./config/network_yolov7/file.yaml
```
进行测试，测试结果将存入`run_test`文件夹。测试过程中的输出情况例如：
```
Test create dataset success, epoch size 78.
Test step 1/78, cost time 0.68s
Test step 2/78, cost time 0.72s
Test step 3/78, cost time 0.64s
```
这里给出自测结果，下面图片分别是混淆矩阵confusion_matrix、查准率和召回率的调和平均数F1-score、准确率和置信度的关系图P_curve、精准率与召回率的关系图PR_curve以及召回率和置信度之间的关系图R_curve。
<img width="550" height="350" src="https://github.com/RintaClio/my_picture/blob/main/confusion_matrix.png?raw=true" alt="混淆矩阵"/>
<img width="550" height="350" src="https://github.com/RintaClio/my_picture/blob/main/F1_curve.png?raw=true" alt="F1-score"/>
<img width="550" height="350" src="https://github.com/RintaClio/my_picture/blob/main/P_curve.png?raw=true" alt="P-curve"/>
<img width="565" height="350" src="https://github.com/RintaClio/my_picture/blob/main/PR_curve.png?raw=true" alt="PR-curve"/>
<img width="550" height="350" src="https://github.com/RintaClio/my_picture/blob/main/R_curve.png?raw=true" alt="R-curve"/>

同时根据日志输出，这里计算并绘制出平均loss`avg_loss`如下。\
<img width="350" height="330" src="https://github.com/RintaClio/my_picture/blob/main/avg_loss.jpg?raw=true" alt="F1-score"/>

此外，可以通过加载训练好的模型对图片进行检测，查看效果。这里利用`detect`文件夹下的`detect.py`，在终端运行如下命令进行目标检测：
```
cd detect
python detect.py
```
待检测图片存放在`detect/detect_images`文件夹，用户可自行增删。检测得到的结果存放在`results`文件夹。这里给出几个检测结果示例。
<img width="450" height="250" src="https://github.com/RintaClio/my_picture/blob/main/1.jpg?raw=true" alt="animals"/>   <img width="450" height="250" src="https://github.com/RintaClio/my_picture/blob/main/animals.jpg?raw=true" alt="animals1"/> \
<img width="450" height="250" src="https://github.com/RintaClio/my_picture/blob/main/2.jpg?raw=true" alt="basketball"/>   <img width="450" height="250" src="https://github.com/RintaClio/my_picture/blob/main/basketball.jpg?raw=true" alt="basketball1"/> \
<img width="180" height="250" src="https://github.com/RintaClio/my_picture/blob/main/3.jpg?raw=true" alt="celebrity"/>   <img width="180" height="250" src="https://github.com/RintaClio/my_picture/blob/main/celebrity.jpg?raw=true" alt="celebrity1"/> \
<img width="400" height="250" src="https://github.com/RintaClio/my_picture/blob/main/4.jpg?raw=true" alt="crowd"/>   <img width="400" height="250" src="https://github.com/RintaClio/my_picture/blob/main/crowd.jpg?raw=true" alt="animals1"/>
<img width="450" height="250" src="https://github.com/RintaClio/my_picture/blob/main/5.jpg?raw=true" alt="party"/>   <img width="450" height="250" src="https://github.com/RintaClio/my_picture/blob/main/party.jpg?raw=true" alt="party1"/>
<img width="450" height="250" src="https://github.com/RintaClio/my_picture/blob/main/6.jpg?raw=true" alt="traffic"/>   <img width="450" height="250" src="https://github.com/RintaClio/my_picture/blob/main/traffic.jpg?raw=true" alt="animals1"/>

这里在`detect`文件夹中附上由`yolov7_tiny_ours.ckpt`模型转换得到的`.pt`格式的模型，用于pytorch框架下代码的目标检测测试。转换方法为：首先，使用`load_checkpoint`读取MindSpore生成的checkpoint文件，拿到参数名和参数值并存入`.txt`文件。然后，根据一个现有的yolov7的`.pth`模型，对照参数名将相应的参数值修改为所读取到的`.ckpt`的值。
其中`conv.weight`相互对应，pth模型的`bn.bias`对应ckpt模型的`bn.beta`，pth模型的`bn.weight`对应ckpt模型的`bn.gamma`，pth模型的`running_mean`对应ckpt模型的`moving_mean`，pth模型的`running_var`对应ckpt模型的`moving_varience`。最后，将`.pth`格式的模型转换为`.pt`格式的模型。

## 成员
SA22010072  张可成 \
SA22010077  周艺嘉
