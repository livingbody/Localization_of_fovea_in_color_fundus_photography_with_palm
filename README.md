#  飞桨常规赛：PALM眼底彩照中黄斑中央凹定位比赛 5月第5名方案
比赛链接： [常规赛：PALM眼底彩照中黄斑中央凹定位](https://aistudio.baidu.com/aistudio/competition/detail/86)

**github链接： [https://github.com/livingbody/Localization_of_fovea_in_color_fundus_photography_with_palm](https://github.com/livingbody/Localization_of_fovea_in_color_fundus_photography_with_palm)**

**aistudio链接： [https://aistudio.baidu.com/aistudio/projectdetail/2027101](https://aistudio.baidu.com/aistudio/projectdetail/2027101)**

## 0.0赛题介绍
PALM黄斑定位常规赛的重点是研究和发展与患者眼底照片黄斑结构定位相关的算法。该常规赛的目标是评估和比较在一个常见的视网膜眼底图像数据集上定位黄斑的自动算法。具体目的是预测黄斑中央凹在图像中的坐标值。

![](https://ai.bdstatic.com/file/D80ACC7A393348DD9F30518D57781D32)

## 0.1数据简介
PALM病理性近视预测常规赛由中山大学中山眼科中心提供800张带黄斑中央凹坐标标注的眼底彩照供选手训练模型，另提供400张带标注数据供平台进行模型测试。

## 0.2数据说明
本次常规赛提供的金标准由中山大学中山眼科中心的7名眼科医生手工进行标注，之后由另一位高级专家将它们融合为最终的标注结果。本比赛提供数据集对应的黄斑中央凹坐标信息存储在xlsx文件中，名为“Fovea_Location_train”，第一列对应眼底图像的文件名(包括扩展名“.jpg”)，第二列包含x坐标，第三列包含y坐标。

![](https://ai.bdstatic.com/file/1CD1DA54E68349CA8553678E80F4D40E)

## 0.3训练数据集
文件名称：Train
Train文件夹里有一个文件夹fundus_images和一个xlsx文件。

fundus_images文件夹内包含800张眼底彩照，分辨率为1444×1444，或2124×2056。命名形如H0001.jpg、P0001.jpg、N0001.jpg和V0001.jpg。
xlsx文件中包含800张眼底彩照对应的x、y坐标信息。
## 0.4测试数据集
文件名称：PALM-Testing400-Images 文件夹里包含400张眼底彩照，命名形如T0001.jpg。

## 0.5提交格式
黄斑中央凹定位比赛的提交内容需将所有测试图像的黄斑坐标存入一个CSV文件，名为“Fovea_Localization_Results.csv”，第一列对应测试眼底图像的文件名(包括扩展名“.jpg”)，第二列包含x坐标，第三列包含y坐标。

![](https://ai.bdstatic.com/file/0D28C4FD24CA47748B867579C9A4CABC)

## 0.6思路
训练数据为图片以及中心点，要想从这些数据中推测测试数据中心店位置信息。可认为构造bbox，然后按目标检测来做，预测新的bbox，然后利用bbox来计算中心点，达到所需信息。

# 1.数据格式调整


```python
!unzip -aqo data/data85130/常规赛：PALM眼底彩照中黄斑中央凹定位.zip
```


```python
!rm _* -rf
```


```python
!mv '常规赛：PALM眼底彩照中黄斑中央凹定位' dataset
```


```python
!mv  dataset/Train/fundus_image  dataset/Train/JPEGImages 
```


```python
import pandas as pd

train_data=pd.read_excel('dataset/Train/Fovea_Location_train.xlsx')
train_data.head()

```

## 1.1 自定义生成VOC数据集格式函数


```python
# lxml操作需要用到
!pip install lxml
```


```python
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os


def save_xml(image_name, bbox, save_dir='Annotations', width=10, height=10, channel=3):
    '''
    将CSV中的一行
    000000001.jpg [[1,2,3,4],...]
    转化成
    000000001.xml

    :param image_name:图片名
    :param bbox:对应的bbox
    :param save_dir:
    :param width:这个是图片的宽度，博主使用的数据集是固定的大小的，所以设置默认
    :param height:这个是图片的高度，博主使用的数据集是固定的大小的，所以设置默认
    :param channel:这个是图片的通道，博主使用的数据集是固定的大小的，所以设置默认
    :return:
    '''

    # 图片存放位置
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    # 文件名
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.join('JPEGImages',image_name.split('/')[-1])

    # 文件长
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    # 文件高
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    # 文件通道
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    # bbox
    for x, y, w, h in bbox:
        print(x, y, w, h)
        left, top, right, bottom = x, y, x + w, y + h
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'bad'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    # 保存xml
    image_name=image_name.split('/')[-1]
    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)
    return
```

## 1.2 开始生成voc格式xml


```python
import cv2
import os
train_base_path='dataset/Train'
# for i in range(10):
for i in range(len(train_data)):
    x=train_data.values[i,1]
    y=train_data.values[i,2]
    bbox=[[float(x-5), float(y-20), 40, 40]]
    # print(bbox)
    image_name=os.path.join(train_base_path,'JPEGImages', train_data.values[i,0])
    img=cv2.imread(image_name)
    # 0--height, 1--widht, 2--channel
    height, width, channel = img.shape
    
    save_xml(image_name, bbox, save_dir=os.path.join(train_base_path,'Annotations'), width=width, height=height, channel=channel)
```

# 2.paddlex安装配置

## 2.1 paddlex安装


```python
!pip install paddlex
```

## 2.2 数据集划分


```python
# 分割训练集、测试集
!paddlex --split_dataset --format VOC --dataset_dir dataset/Train --val_value 0.2
```

# 3.paddlex数据集配置

## 3.1引入相关包


```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex.det import transforms
```

## 3.2 Transform


```python
# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/seg_transforms.html
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32),
])
```

## 3.3 数据集


```python
# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-changedetdataset
train_dataset = pdx.datasets.VOCDetection(
    data_dir='dataset/Train',
    file_list='dataset/Train/train_list.txt',
    label_list='dataset/Train/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='dataset/Train',
    file_list='dataset/Train/val_list.txt',
    label_list='dataset/Train/labels.txt',
    transforms=eval_transforms)
```

## 3.4 模型调用
注意不同模型，背景是否也算作一类，也是有说法的。


```python
# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels) + 1


# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3
model = pdx.det.FasterRCNN(num_classes=num_classes)
```

# 4.开始训练
我也很无语，40*40的框，就这么点精度

```
2021-05-26 20:39:03 [INFO]	[EVAL] Finished, Epoch=17, bbox_map=0.002684 .
2021-05-26 20:39:07 [INFO]	Model saved in output/faster_rcnn_r50_fpn/epoch_17.
2021-05-26 20:39:07 [INFO]	Current evaluated best model in eval_dataset is epoch_8, bbox_map=4.625950541327286
```


```python
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=26,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    lr_decay_epochs=[210, 240],
    save_dir='output/faster_rcnn_r50_fpn',    
    use_vdl=True)
```

# 5.模型预测

## 5.1 先预测一张


```python
import paddlex as pdx
model = pdx.load_model('output/faster_rcnn_r50_fpn/best_model')
image_name = 'dataset/PALM-Testing400-Images/T0001.jpg'
result = model.predict(image_name)
print(result)
pdx.det.visualize(image_name, result, threshold=0.5, save_dir='./output/faster_rcnn_r50_fpn')
```


```python
bbox=result[0]['bbox']
print(bbox)
```

## 5.2 生成文件名列表


```python
# 预测数据集val_list
val_list=[]
for i in range(1,401,1):
    filename='T'+ str(i).zfill(4)+'.jpg'
    # print(filename)
    val_list.append(filename+'\n')

with open('val_list.txt','w') as f:
    f.writelines(val_list)
```


```python
pd_A=[]
with open('val_list.txt', 'r') as f:
    for line in f:
        line='dataset/PALM-Testing400-Images/'+line
        pd_A.append(line.split('\n')[0])
```

## 5.3预测并生成x/y坐标


```python
import paddlex as pdx

model = pdx.load_model('output/faster_rcnn_r50_fpn/best_model')
# x坐标
pd_B=[]
# y坐标
pd_C=[]
for item in pd_A :
    result = model.predict(item)
    if  result:
        bbox=result[0]['bbox']
        x0=bbox[0]
        x1=bbox[2]
        y0=bbox[1]
        y1=bbox[3]
        x=0.5*(x1+x0)
        y=0.5*(y1+y0)
    else:
        x=0
        y=0
    pd_B.append(x)
    pd_C.append(y)

```

## 5.4 构造pandas framework


```python
# 文件名列
pd_A=[]
with open('val_list.txt', 'r') as f:
    for line in f:
        pd_A.append(line.split('\n')[0])
```


```python
# 构造pandas的DataFrame
df= pd.DataFrame({'FileName': pd_A, 'Fovea_x':pd_B, 'Fovea_y':pd_C})
```

## 5.5 保存结果


```python
# 保存为提交文件
df.to_csv("Fovea_Localization_Results.csv", index=None)
```


```python
# 读取检查下
data=pd.read_csv('Fovea_Localization_Results.csv')
```


```python
# 查看下生成数据
data.head(20)
```

# 6.提交结果


![](https://ai-studio-static-online.cdn.bcebos.com/7a8e9138690047d7b021e99d91fa633eff9a8e76e157476bb4ef91fa64c67989)

## 6.1 一个是要注意分割类别，不同模型是否要加背景，需要特别注意
## 6.2 框的大小选择需要慎重，需要切合实际
## 6.3 框的大小后来测试80左右最佳


