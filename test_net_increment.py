# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.vgg16_split_clsbox_test import vgg16_split_clsbox_test
from model.faster_rcnn.resnet import resnet
#from model.faster_rcnn.resnet_split_clsbox_test import resnet_split_clsbox_test
from model.faster_rcnn.resnet_split_clsbox_test_residual import resnet_split_clsbox_test
from model.utils.config import cfg
#from model.faster_rcnn.resnet_split_clsbox_test_wa import resnet_split_clsbox_test_wa as resnet_split_clsbox_test   # weight algin post process
#from model.faster_rcnn.resnet_split_clsbox_test_v2 import resnet_split_clsbox_test
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc_test', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='vgg16', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--residual_model', dest='residual_model',
                      help='directory to load models', default="",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=20, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=1747, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--save', dest='save',
                      help='save feature mode',
                      default='',
                      type=str)
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test_base":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_base"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_total_add"
      args.imdb_name_expert_10 = "voc_2007_10_train_expert"
      args.imdb_name_expert_5 = "voc_2007_5_train_expert"
      args.imdb_name_expert_1 = "voc_2007_1_train_tv"  ########## last new class
      args.imdb_name_org_19 = 'voc_2007_trainval'
      args.imdb_name_org_10 = 'voc_2007_10_train'
      args.imdb_name_org_15 = 'voc_2007_15_train'
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test_plant_change":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_plant_change"
      args.imdb_name_expert = "voc_2007_19_1_plant_inc"  ########## last new class
      args.imdb_name_org = 'voc_2007_19_ex_plant'
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test_sheep_change":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sheep_change"
      args.imdb_name_expert = "voc_2007_19_1_sheep_inc"  ########## last new class
      args.imdb_name_org = 'voc_2007_19_ex_sheep'
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test_sofa_change":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sofa_change"
      args.imdb_name_expert = "voc_2007_19_1_sofa_inc"  ########## last new class
      args.imdb_name_org = 'voc_2007_19_ex_sofa'
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test_train_change":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_train_change"
      args.imdb_name_expert = "voc_2007_19_1_train_inc"  ########## last new class
      args.imdb_name_org = 'voc_2007_19_ex_train'
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test_sqe":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test_sqe_plant":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_plant"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_plant"
      args.imdb_name_org = 'voc_2007_15_train'
  elif args.dataset == "pascal_voc_test_sqe_sheep":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_sheep"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_sheep"
      args.imdb_name_org = 'voc_2007_15_plant'
  elif args.dataset == "pascal_voc_test_sqe_sofa":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_sofa"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_sofa"
      args.imdb_name_org = 'voc_2007_16_sheep'
  elif args.dataset == "pascal_voc_test_sqe_train":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_train"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_train"
      args.imdb_name_org = 'voc_2007_17_sofa'
  elif args.dataset == "pascal_voc_test_sqe_tv":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_tv"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_tv"
      args.imdb_name_org = 'voc_2007_18_train'
  elif args.dataset == "pascal_voc_07_15":
      args.imdb_name = "voc_2007_15_train"
      args.imdbval_name = "voc_2007_15_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_org = 'coco_40_train_base'
      args.imdb_name_expert = "coco_40_train_expert"
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_test_sqe_table":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_table"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_table"
      args.imdb_name_org = 'voc_2007_10_train'
  elif args.dataset == "pascal_voc_test_sqe_dog":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_dog"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_dog"
      args.imdb_name_org = 'voc_2007_10_10_table'
  elif args.dataset == "pascal_voc_test_sqe_horse":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_horse"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_horse"
      args.imdb_name_org = 'voc_2007_10_11_dog'
  elif args.dataset == "pascal_voc_test_sqe_motorbike":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_motorbike"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_motorbike"
      args.imdb_name_org = 'voc_2007_10_12_horse'
  elif args.dataset == "pascal_voc_test_sqe_person":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_sqe_person"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_1_train_person"
      args.imdb_name_org = 'voc_2007_10_13_motorbike'
  elif args.dataset == "pascal_voc_test_sqe_5b":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_5_b"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_5_b"
      args.imdb_name_org = 'voc_2007_5_a'
  elif args.dataset == "pascal_voc_test_sqe_5c":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_5_c"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_5_c"
      args.imdb_name_org = 'voc_2007_5_b'
  elif args.dataset == "pascal_voc_test_sqe_5d":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test_5_d"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "voc_2007_5_d"
      args.imdb_name_org = 'voc_2007_5_c'
  elif args.dataset == "coco_test_b":
      args.imdb_name = "coco_b"
      args.imdbval_name = "coco_test_b"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "coco_b_expert"
      args.imdb_name_org = 'coco_a'
  elif args.dataset == "coco_test_c":
      args.imdb_name = "coco_c"
      args.imdbval_name = "coco_test_c"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "coco_c_expert"
      args.imdb_name_org = 'coco_b'
  elif args.dataset == "coco_test_d":
      args.imdb_name = "coco_d"
      args.imdbval_name = "coco_test_d"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
      args.imdb_name_expert = "coco_d_expert"
      args.imdb_name_org = 'coco_c'

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  print(cfg.NEW_CLASSES)
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  if cfg.NEW_CLASSES==5 and 'sqe' not in args.dataset:
      imdb_expert_cls=args.imdb_name_expert_5
      imdb_org_name=args.imdb_name_org_15
  elif cfg.NEW_CLASSES==10 and 'sqe' not in args.dataset:
      imdb_expert_cls=args.imdb_name_expert_10
      imdb_org_name = args.imdb_name_org_10
  elif cfg.NEW_CLASSES==1 and args.dataset == "pascal_voc_test":
      imdb_expert_cls=args.imdb_name_expert_1
      imdb_org_name = args.imdb_name_org_19
  elif 'pascal_voc_test_sqe_' in args.dataset:
      imdb_expert_cls=args.imdb_name_expert
      imdb_org_name=args.imdb_name_org
  elif 'coco' in args.dataset:
      imdb_expert_cls = args.imdb_name_expert
      imdb_org_name = args.imdb_name_org
  else:
      imdb_expert_cls=args.imdb_name_expert
      imdb_org_name=args.imdb_name_org
  imdb_expert, roidb_expert, ratio_list_expert, ratio_index_expert = combined_roidb(imdb_expert_cls, False)
  imdb_org, roidb_org, ratio_list_org, ratio_index_org = combined_roidb(imdb_org_name, False)

  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir#+ "/" + args.net + "/" + 'pascal_voc_0712_incre'
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = args.load_dir #os.path.join(input_dir,
   # 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN = vgg16_split_clsbox_test(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)###############split old and new cls
  elif args.net == 'res101':
    #fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN_residual = resnet(imdb_expert.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN = resnet_split_clsbox_test(imdb.classes, 101, pretrained=False,
                                          class_agnostic=args.class_agnostic)  ###############split old and new cls
  elif args.net == 'res50':
    fasterRCNN_residual = resnet(imdb_expert.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN = resnet_split_clsbox_test(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)###############split old and new cls
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    #pdb.set_trace()

  fasterRCNN.create_architecture()
  fasterRCNN_residual.create_architecture()
  '''
  ############### the parameters of base class of inc model are initialized by base model ##########################
  base_model_19 = 'model_save_dir/models_res50_caffe_voc07_19/res50/pascal_voc/faster_rcnn_1_20_9873.pth'  # 'models_res50_voc19/res50/pascal_voc_0712/faster_rcnn_1_18_32597.pth'
  base_model_15 = 'model_save_dir/models_res50_voc15_new/res50/pascal_voc_07_15/faster_rcnn_1_20_9003.pth'
  base_model_10 = 'model_save_dir/models_res50_voc10/res50/pascal_voc_07_10/faster_rcnn_1_20_6003.pth'
  base_model_coco = 'model_save_dir/models_res50_coco40/res50/coco_40_train/faster_rcnn_1_20_34040.pth'
  if cfg.NEW_CLASSES==5:
      base_model=base_model_15
  elif cfg.NEW_CLASSES==10:
      base_model=base_model_10
  elif cfg.NEW_CLASSES==1:
      base_model=base_model_19
  #base_model=base_model_15
  if base_model:
      #base_model = args.base_model
      if args.cuda > 0:
          checkpoint_base = torch.load(base_model)
      else:
          checkpoint_base = torch.load(base_model, map_location=(lambda storage, loc: storage))
      pretrained_dict_base = {k: v for k, v in
                              checkpoint_base['model'].items() if 'RCNN_base' in k}
      
      ################### feat=(inc+(org+res))/2 ####################################################
      #fasterRCNN_org = resnet(imdb_org.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
      #fasterRCNN_org.create_architecture()
      #fasterRCNN_org.load_state_dict(checkpoint_base['model'])
      ###############################################################################################
  ###################################################################################################################
   '''




  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  '''
  load_name_residual=args.residual_model
  checkpoint_res = torch.load(load_name_residual)
  fasterRCNN_residual.load_state_dict(checkpoint_res['model'])
  '''
  frcnn_dict = fasterRCNN.state_dict()
  pretrained_dict = {k: v for k, v in checkpoint['model'].items() if 'embedding' not in k}
  frcnn_dict.update(pretrained_dict)

  #frcnn_dict.update(pretrained_dict_base) ######### initialize the backbone with old model : fea=org+res

  fasterRCNN.load_state_dict(frcnn_dict)#(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()
    #fasterRCNN_residual.cuda()
    #fasterRCNN_org.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.0 #0.05
  else:
    thresh = 0.0#0.0

  save_name = 'faster_rcnn_incre_voc07test'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  #fasterRCNN_residual.eval()
  #fasterRCNN_org.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  save_fea_flag=args.save
  if save_fea_flag:
    thresh=0.9
    pooled_feat_label=torch.Tensor([]).cuda()
    pooled_feat_all=torch.Tensor([]).cuda()

  for i in range(num_images):

      data = next(data_iter)
      with torch.no_grad():
          im_data.resize_(data[0].size()).copy_(data[0])
          im_info.resize_(data[1].size()).copy_(data[1])
          gt_boxes.resize_(data[2].size()).copy_(data[2])
          num_boxes.resize_(data[3].size()).copy_(data[3])

          det_tic = time.time()
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label,pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,fasterRCNN_residual=None,fasterRCNN_org=None)


      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)



      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            pooled_feat_t = pooled_feat[inds]
            _, order = torch.sort(cls_scores, 0, True)

            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            pooled_feat_t = pooled_feat_t[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            pooled_feat_t = pooled_feat_t[keep.view(-1).long()]
            if save_fea_flag and pooled_feat_label.shape[0]==0:
                pooled_feat_label = torch.Tensor([int(j)-1]).repeat(pooled_feat_t.shape[0]).to(torch.int32)
                pooled_feat_all = pooled_feat_t
            elif save_fea_flag:
                pooled_feat_label = torch.cat((pooled_feat_label, torch.Tensor([int(j)-1]).repeat(pooled_feat_t.shape[0]).to(torch.int32)),dim=0)
                pooled_feat_all = torch.cat((pooled_feat_all, pooled_feat_t),dim=0)
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.85)#0.3
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array





      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          save_dir='vis10/'
          if not os.path.exists(save_dir):
              os.mkdir(save_dir)
          print(save_dir+imdb.image_path_at(i).split('/')[-1])
          cv2.imwrite(save_dir+imdb.image_path_at(i).split('/')[-1], im2show)
          #pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)
  if save_fea_flag:
    torch.save(pooled_feat_all, 'res_fea_'+save_fea_flag+'.pth')
    torch.save(pooled_feat_label, 'res_label_'+save_fea_flag+'.pth')
  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
