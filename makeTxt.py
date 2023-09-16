import os
import random

trainval_percent = 0.3
train_percent = 0.7
xmlfilepath = '/home/yuan3080/桌面/detection_paper_6/mmdetection-master1/mmdetection-master/VOCdevkit/VOC2007/Annotations'
txtsavepath = '/home/yuan3080/桌面/detection_paper_6/mmdetection-master1/mmdetection-master/VOCdevkit/VOC2007/JPEGImages'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')
ftest = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
ftrain = open('VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
fval = open('VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
