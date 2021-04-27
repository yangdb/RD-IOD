# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_incre import pascal_voc_incre
from datasets.pascal_voc_incre_add import pascal_voc_incre_add
from datasets.pascal_voc_incre_add_new import pascal_voc_incre_add_new
from datasets.pascal_voc_test import pascal_voc_test
from datasets.coco import coco
from datasets.coco_inc import coco_inc
from datasets.coco_sqe import coco_sqe
from datasets.imagenet import imagenet
from datasets.vg import vg

from datasets.pascal_voc_07_15 import pascal_voc_07_15
from datasets.pascal_voc_07_10 import pascal_voc_07_10
from datasets.pascal_voc_07_15_test import pascal_voc_07_15_test

from datasets.coco_14_40_base import coco_14_40_base
from datasets.pascal_voc_07_15_test import pascal_voc_07_15_test
from datasets.coco_test import coco_test

from datasets.pascal_voc_inc_sqe import pascal_voc_inc_sqe
from datasets.pascal_voc_inc_sqe_test import pascal_voc_inc_sqe_test
from datasets.pascal_voc_07_10_test import pascal_voc_07_10_test
import numpy as np

data_path_ydb='/mnt/disk7/ydb/Object_train/faster-rcnn.pytorch/'
# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc_incre(split, year))
        __sets[name+'_incre'] = (lambda split=split, year=year: pascal_voc_incre_add(split, year,devkit_path=data_path_ydb+'data/'+'VOCdevkit'+year+'-1'))
__sets['voc_2007_test_total'] = (lambda split='test', year='2007': pascal_voc_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest'))
__sets['voc_2007_test_total_add'] = (lambda split='test', year='2007': pascal_voc_incre_add(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest'))
__sets['voc_2007_test_base'] = (lambda split='test', year='2007': pascal_voc_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit'))
__sets['voc_2007_trainval_all'] = (lambda split='trainval', year='2007': pascal_voc_incre_add(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit'))
__sets['voc_2012_trainval_all'] = (lambda split='trainval', year='2012': pascal_voc_incre_add(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit'))
__sets['voc_2007_15_train'] = (lambda split='trainval', year='2007': pascal_voc_07_15(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-15/'))
__sets['voc_2007_15_test'] = (lambda split='test', year='2007': pascal_voc_07_15_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest'))
__sets['voc_2007_5_incre'] = (lambda split='trainval', year='2007': pascal_voc_incre_add_new(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-5/'))

__sets['voc_2007_10_train'] = (lambda split='trainval', year='2007': pascal_voc_07_10(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10/'))
__sets['voc_2007_10_test'] = (lambda split='test', year='2007': pascal_voc_07_10_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest'))
__sets['voc_2007_10_test_ex'] = (lambda split='test', year='2007': pascal_voc_07_10_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10/',ex=True))
__sets['voc_2007_15_test_ex'] = (lambda split='test', year='2007': pascal_voc_07_15_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-15/',ex=True))
__sets['voc_2007_19_test_ex'] = (lambda split='test', year='2007': pascal_voc_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-19/',ex=True))
__sets['voc_2007_10_incre'] = (lambda split='trainval', year='2007': pascal_voc_incre_add_new(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc/'))

__sets['voc_2007_19_ex_plant'] = (lambda split='trainval', year='2007': pascal_voc_07_15(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-19-ex_plant/',cls19=['sheep', 'sofa', 'train', 'tvmonitor']))
__sets['voc_2007_19_ex_plant_test'] = (lambda split='trainval', year='2007': pascal_voc_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['sheep', 'sofa', 'train', 'tvmonitor']))
__sets['voc_2007_19_ex_sheep'] = (lambda split='trainval', year='2007': pascal_voc_07_15(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-19-ex_sheep/',cls19=[ 'pottedplant', 'sofa', 'train', 'tvmonitor']))
__sets['voc_2007_19_ex_sheep_test'] = (lambda split='trainval', year='2007': pascal_voc_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=[ 'pottedplant', 'sofa', 'train', 'tvmonitor']))
__sets['voc_2007_19_ex_sofa'] = (lambda split='trainval', year='2007': pascal_voc_07_15(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-19-ex_sofa/',cls19=[ 'pottedplant','sheep', 'train', 'tvmonitor']))
__sets['voc_2007_19_ex_sofa_test'] = (lambda split='trainval', year='2007': pascal_voc_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=[ 'pottedplant','sheep', 'train', 'tvmonitor']))
__sets['voc_2007_19_ex_train'] = (lambda split='trainval', year='2007': pascal_voc_07_15(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-19-ex_train/',cls19=[ 'pottedplant','sheep', 'sofa', 'tvmonitor']))
__sets['voc_2007_19_ex_train_test'] = (lambda split='trainval', year='2007': pascal_voc_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=[ 'pottedplant','sheep', 'sofa', 'tvmonitor']))

__sets['voc_2007_19_plant_inc'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-pottedplant/',clas=['sheep', 'sofa', 'train','tvmonitor','pottedplant']))
__sets['voc_2007_19_sheep_inc'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sheep/',clas=['pottedplant', 'sofa', 'train','tvmonitor', 'sheep']))
__sets['voc_2007_19_sofa_inc'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sofa/',clas=['pottedplant', 'sheep', 'train','tvmonitor', 'sofa']))
__sets['voc_2007_19_train_inc'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-train/',clas=['pottedplant', 'sheep', 'sofa','tvmonitor', 'train']))

__sets['voc_2007_test_plant_change'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['sheep', 'sofa', 'train','tvmonitor','pottedplant']))
__sets['voc_2007_test_sheep_change'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant', 'sofa', 'train','tvmonitor', 'sheep']))
__sets['voc_2007_test_sofa_change'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant', 'sheep', 'train','tvmonitor', 'sofa']))
__sets['voc_2007_test_train_change'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep', 'sofa','tvmonitor','train']))

__sets['voc_2007_19_1_plant_inc'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-pottedplant/',clas=['pottedplant'],onlytrain=True))
__sets['voc_2007_19_1_sheep_inc'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sheep/',clas=['sheep'],onlytrain=True))
__sets['voc_2007_19_1_sofa_inc'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sofa/',clas=['sofa'],onlytrain=True))
__sets['voc_2007_19_1_train_inc'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-train/',clas=['train'],onlytrain=True))

for inc_cls in ['diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']:
    __sets['voc_2007_10_inc_'+inc_cls] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-'+inc_cls+'/',clas=inc_cls))



__sets['voc_2007_1_train_table'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-diningtable/',clas=['diningtable'],onlytrain=True))
__sets['voc_2007_1_train_dog'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-dog/',clas=['dog'],onlytrain=True))
__sets['voc_2007_1_train_horse'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-horse/',clas=['horse'],onlytrain=True))
__sets['voc_2007_1_train_motorbike'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-motorbike/',clas=['motorbike'],onlytrain=True))
__sets['voc_2007_1_train_person'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-person/',clas=['person'],onlytrain=True))

__sets['voc_2007_1_train_plant'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-pottedplant/',clas=['pottedplant'],onlytrain=True))#2007-10-inc-one-gt#2007-10-inc-one
__sets['voc_2007_1_train_sheep'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sheep/',clas=['sheep'],onlytrain=True))#2007-10-inc-one-gt#2007-10-inc-one
__sets['voc_2007_1_train_sofa'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sofa/',clas=['sofa'],onlytrain=True))#2007-10-inc-one-gt#2007-10-inc-one
__sets['voc_2007_1_train_train'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-train/',clas=['train'],onlytrain=True))#2007-10-inc-one-gt#2007-10-inc-one
__sets['voc_2007_1_train_tv'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-tvmonitor/',clas=['tvmonitor'],onlytrain=True))#2007-10-inc-one-gt#2007-10-inc-one
#pottedplant,'sheep','sofa','train','tvmonitor'

__sets['voc_2007_5_train_expert'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-5-expert/',clas=['pottedplant', 'sheep', 'sofa', 'train','tvmonitor'],onlytrain=True))#2007-10-inc-one-gt#2007-10-inc-one
__sets['voc_2007_10_train_expert'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-expert/',clas=['diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant', 'sheep', 'sofa', 'train','tvmonitor'],onlytrain=True))#2007-10-inc-one-gt#2007-10-inc-one


__sets['voc_2007_15_plant'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-pottedplant/',clas=['pottedplant']))#2007-10-inc-one-gt#2007-10-inc-one
#['pottedplant','sheep','sofa','train','tvmonitor']
#__sets['voc_2007_test_sqe'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep','sofa','train','tvmonitor']))######### class increase
__sets['voc_2007_16_sheep'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sheep/',clas=['pottedplant','sheep']))
__sets['voc_2007_17_sofa'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sofa/',clas=['pottedplant','sheep','sofa']))
__sets['voc_2007_18_train'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-train/',clas=['pottedplant','sheep','sofa','train']))
__sets['voc_2007_19_tv'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-tvmonitor/',clas=['pottedplant','sheep','sofa','train','tvmonitor']))


base_cls=['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair','cow']
__sets['voc_2007_10_10_table'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-diningtable/',base=base_cls,clas=['diningtable']))
__sets['voc_2007_10_11_dog'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-dog/',base=base_cls,clas=['diningtable','dog']))
__sets['voc_2007_10_12_horse'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-horse/',base=base_cls,clas=['diningtable','dog','horse']))
__sets['voc_2007_10_13_motorbike'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-motorbike/',base=base_cls,clas=['diningtable','dog','horse','motorbike']))
__sets['voc_2007_10_14_person'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-person/',base=base_cls,clas=['diningtable','dog','horse','motorbike','person']))
__sets['voc_2007_10_15_pottedplant'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-pottedplant/',base=base_cls,clas=['diningtable','dog','horse','motorbike','person','pottedplant']))
__sets['voc_2007_10_16_sheep'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sheep/',base=base_cls,clas=['diningtable','dog','horse','motorbike','person','pottedplant','sheep']))
__sets['voc_2007_10_17_sofa'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-sofa/',base=base_cls,clas=['diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa']))
__sets['voc_2007_10_18_train'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-train/',base=base_cls,clas=['diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train']))
__sets['voc_2007_10_19_tvmonitor'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-10-inc-one/inc-tvmonitor/',base=base_cls,clas=['diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']))

base_cls_a=['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle']
'''
__sets['voc_2007_5_b'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-5-b/',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow']))
__sets['voc_2007_5_c'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-5-c/',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow','diningtable','dog','horse','motorbike','person']))
__sets['voc_2007_5_d'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-5-d/',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']))
__sets['voc_2007_5_a'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/2007-5-a/',base=base_cls_a))
'''



__sets['voc_2007_5_b'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/VOC2007/',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow'],img_set='trainvalStep1.txt'))
__sets['voc_2007_5_c'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/VOC2007/',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow','diningtable','dog','horse','motorbike','person'],img_set='trainvalStep2.txt'))
__sets['voc_2007_5_d'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/VOC2007/',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'],img_set='trainvalStep3.txt'))
__sets['voc_2007_5_a'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/VOC2007/',base=base_cls_a,img_set='trainvalStep0.txt'))


__sets['coco_a'] = (lambda split='train', year='2017': coco_sqe(split, year,img_set='trainvalStep0.json'))
__sets['coco_b'] = (lambda split='train', year='2017': coco_sqe(split, year,img_set='trainvalStep1.json',inc='trainvalStep1a.json'))
__sets['coco_c'] = (lambda split='train', year='2017': coco_sqe(split, year,img_set='trainvalStep2.json',inc='trainvalStep2a.json'))
__sets['coco_d'] = (lambda split='train', year='2017': coco_sqe(split, year,img_set='trainvalStep3.json',inc='trainvalStep3a.json'))
__sets['coco_b_expert'] = (lambda split='train', year='2017': coco_sqe(split, year,img_set='trainvalStep1.json'))
__sets['coco_c_expert'] = (lambda split='train', year='2017': coco_sqe(split, year,img_set='trainvalStep2.json'))
__sets['coco_d_expert'] = (lambda split='train', year='2017': coco_sqe(split, year,img_set='trainvalStep3.json'))

__sets['coco_test_a'] = (lambda split='val', year='2017': coco_sqe(split, year,img_set='testStep0a.json'))
__sets['coco_test_b'] = (lambda split='val', year='2017': coco_sqe(split, year,img_set='testStep1a.json'))
__sets['coco_test_c'] = (lambda split='val', year='2017': coco_sqe(split, year,img_set='testStep2a.json'))
__sets['coco_test_d'] = (lambda split='val', year='2017': coco_sqe(split, year,img_set='testStep3a.json'))



__sets['voc_2007_1_5b'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/VOC2007/',clas=['bus', 'car', 'cat', 'chair','cow'],onlytrain=True,img_set='trainvalStep1.txt'))#2007-10-inc-one-gt#2007-10-inc-one
__sets['voc_2007_1_5c'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/VOC2007/',clas=['diningtable','dog','horse','motorbike','person'],onlytrain=True,img_set='trainvalStep2.txt'))#2007-10-inc-one-gt#2007-10-inc-one
__sets['voc_2007_1_5d'] = (lambda split='trainval', year='2007': pascal_voc_inc_sqe(split, year,devkit_path=data_path_ydb+'data/'+'VOCdevkit/VOC2007/',clas=['pottedplant','sheep','sofa','train','tvmonitor'],onlytrain=True,img_set='trainvalStep3.txt'))#2007-10-inc-one-gt#2007-10-inc-one



__sets['voc_2007_test_sqe_table'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=base_cls,clas=['diningtable']))#,'sofa','train','tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_dog'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=base_cls,clas=['diningtable','dog']))#,'sofa','train','tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_horse'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=base_cls,clas=['diningtable','dog','horse']))#,'train','tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_motorbike'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=base_cls,clas=['diningtable','dog','horse','motorbike']))#,'tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_person'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=base_cls,clas=['diningtable','dog','horse','motorbike','person']))#,,'sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase


#['diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
__sets['voc_2007_test_sqe'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep']))#,'sofa','train','tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_plant'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant']))#,'sofa','train','tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_sheep'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep']))#,'sofa','train','tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_sofa'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep','sofa']))#,'train','tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_train'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep','sofa','train']))#,'tvmonitor','sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_tv'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep','sofa','train','tvmonitor']))#,,'sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase

__sets['voc_2007_test_sqe_1_sheep'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['sheep'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_1_plant'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_1_sofa'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['sofa'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_1_train'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['train'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_sqe_1_tv'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['tvmonitor'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase

__sets['voc_2007_test_5_expert'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep','sofa','train','tvmonitor'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_10_expert'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant','sheep','sofa','train','tvmonitor'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase

__sets['voc_2007_test_5_a'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test('testStep0a', year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_5_b'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test('testStep1a', year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow']))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_5_c'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test('testStep2a', year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow','diningtable','dog','horse','motorbike','person']))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_5_d'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test('testStep3a', year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=base_cls_a,clas=['bus', 'car', 'cat', 'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']))#))#,,'dog''horse','motorbike','person','plant']))######### class increase

__sets['voc_2007_test_5_a_e'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test('testStep0', year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_5_b_e'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test('testStep1', year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=None,clas=['bus', 'car', 'cat', 'chair','cow'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_5_c_e'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test('testStep2', year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=None,clas=['diningtable','dog','horse','motorbike','person'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase
__sets['voc_2007_test_5_d_e'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test('testStep3', year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',base=None,clas=['pottedplant','sheep','sofa','train','tvmonitor'],onlytrain=True))#))#,,'dog''horse','motorbike','person','plant']))######### class increase



__sets['voc_2007_test_sqe_5b'] = (lambda split='test', year='2007': pascal_voc_inc_sqe_test(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkittest',clas=['pottedplant','sheep','sofa','train','tvmonitor']))#,,'sofa','tvmonitor'))#,,'dog''horse','motorbike','person','plant']))######### class increase


__sets['coco_14_40_train'] = (lambda split='train', year='2014': coco_14_40_base(split, year, devkit_path=data_path_ydb+'data/'+'VOCdevkit/coco_2014/base_40/'))
__sets['coco_40_train_base'] = (lambda split='train', year='2014': coco(split, year,mode='base'))
__sets['coco_40_train_expert'] = (lambda split='train', year='2014': coco(split, year,mode='sp'))
__sets['coco_40_train_inc'] = (lambda split='train', year='2014': coco(split, year,mode='inc'))
__sets['coco_14_train_a'] = (lambda split='train', year='2014': coco(split, year,mode='g_a'))
__sets['coco_14_train_b'] = (lambda split='train', year='2014': coco(split, year,mode='g_b'))
__sets['coco_14_train_c'] = (lambda split='train', year='2014': coco(split, year,mode='g_c'))
__sets['coco_14_train_d'] = (lambda split='train', year='2014': coco(split, year,mode='g_d'))
__sets['coco_14_train_a_expert'] = (lambda split='train', year='2014': coco(split, year,mode='g_a_sp'))
__sets['coco_14_train_b_expert'] = (lambda split='train', year='2014': coco(split, year,mode='g_b_sp'))
__sets['coco_14_train_c_expert'] = (lambda split='train', year='2014': coco(split, year,mode='g_c_sp'))
__sets['coco_14_train_d_expert'] = (lambda split='train', year='2014': coco(split, year,mode='g_d_sp'))

__sets['coco_40_test'] = (lambda split='val', year='2014': coco_test(split, year))
__sets['coco_14_40_incre'] = (lambda split='train', year='2014': coco_inc(split, year))

####### pascal_voc_test:test base classes using all test images(exclude new annotations in test images); pascal_voc_incre_add: test base and new class using all test images;
# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
    for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version, split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))

# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (
        lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split, devkit_path, data_path))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
