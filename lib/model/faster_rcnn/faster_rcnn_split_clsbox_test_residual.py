import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN_split_clsbox_test(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN_split_clsbox_test, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_new_class = cfg.NEW_CLASSES#1#5#5#5#1#10#5 ############################# inc class num
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, fasterRCNN_residual=None,fasterRCNN_org=None):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        #base_feat_res = fasterRCNN_residual.RCNN_base(im_data)
        #base_feat_org = fasterRCNN_org.RCNN_base(im_data)
        #base_feat_add = (base_feat+(base_feat_org+base_feat_res))/2#base_feat #+ base_feat_res

        base_feat_add=base_feat

        #from model.faster_rcnn import show_fea
        #show_fea.save_fea(base_feat)


        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_add, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            # print(gt_boxes.shape)
            if gt_boxes.shape[0]>1:
                gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
                gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
                rois = gt_boxes_append[:, :num_boxes]

                rois_label = Variable(gt_boxes[:,:num_boxes,-1].view(-1).long())

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat_add, rois.view(-1, 5))
            #pooled_feat_inc = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            #pooled_feat_org =fasterRCNN_org.RCNN_roi_align(base_feat_org, rois.view(-1, 5))
            #pooled_feat_residual =fasterRCNN_residual.RCNN_roi_align(base_feat_res, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat_add, rois.view(-1,5))
            #pooled_feat_inc = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            #pooled_feat_org = fasterRCNN_org.RCNN_roi_align(base_feat_org, rois.view(-1, 5))
            #pooled_feat_residual = fasterRCNN_residual.RCNN_roi_align(base_feat_res, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)


        # compute bbox offset
        bbox_pred_old = self.RCNN_bbox_pred(pooled_feat)
        ################# split bbox pred (old and new) ###################
        bbox_pred_new = self.RCNN_bbox_pred_new(pooled_feat)
        bbox_pred_cat = torch.cat((bbox_pred_old, bbox_pred_new), dim=1)
        bbox_pred = bbox_pred_cat
        ###################################################################

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        ################# split score fc (old and new)#####################
        cls_score_new = self.RCNN_cls_score_new(pooled_feat)
        cls_score_cat = torch.cat((cls_score, cls_score_new), dim=1)
        cls_prob = F.softmax(cls_score_cat, 1)
        ###################################################################

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, pooled_feat

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
