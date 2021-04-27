from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

#from model.faster_rcnn.discriminator import Discriminator

class _RPN_distil(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN_distil, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = torch.Tensor([0]).cuda()
        self.rpn_loss_box = torch.Tensor([0]).cuda()

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, fasterRCNN_org=None):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)


        '''
        #################### cat old and new rpn pred #############################
        rpn_cls_prob=torch.cat((rpn_cls_prob_ori,rpn_cls_prob),dim=1)
        rpn_bbox_pred = torch.cat((rpn_bbox_pred_ori, rpn_bbox_pred), dim=1)
        rpn_cls_score = torch.cat((rpn_cls_score_ori,rpn_cls_score),dim=1)
        rpn_cls_score_reshape = torch.cat((rpn_cls_score_reshape_ori, rpn_cls_score_reshape), dim=1)
        ###########################################################################
        '''

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = torch.Tensor([0]).cuda()
        self.rpn_loss_box = torch.Tensor([0]).cuda()

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

            if fasterRCNN_org:
                ################# ori fasterrcnn #####################################################
                rpn_conv1_ori = F.relu(fasterRCNN_org.RCNN_rpn.RPN_Conv(base_feat), inplace=True)
                '''
                rpn_cls_score_ori = fasterRCNN_org.RCNN_rpn.RPN_cls_score(rpn_conv1_ori)
                rpn_cls_score_reshape_ori = fasterRCNN_org.RCNN_rpn.reshape(rpn_cls_score_ori, 2)
                rpn_cls_prob_reshape_ori = F.softmax(rpn_cls_score_reshape_ori, 1)
                rpn_cls_prob_ori = fasterRCNN_org.RCNN_rpn.reshape(rpn_cls_prob_reshape_ori, fasterRCNN_org.RCNN_rpn.nc_score_out)
                rpn_bbox_pred_ori = fasterRCNN_org.RCNN_rpn.RPN_bbox_pred(rpn_conv1_ori)
                cfg_key_ori = 'TRAIN'

                rois_ori = fasterRCNN_org.RCNN_rpn.RPN_proposal((rpn_cls_prob_ori.data, rpn_bbox_pred_ori.data,
                                     im_info, cfg_key_ori))
                ############################################################################################
                
                ############################## distil rpn cls and bbox reg loss ##########################################
                rpn_data_ori = fasterRCNN_org.RCNN_rpn.RPN_anchor_target((rpn_cls_score_ori.data, gt_boxes, im_info, num_boxes))
                rpn_cls_score_ori = rpn_cls_score_reshape_ori.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
                rpn_label_ori = rpn_data_ori[0].view(batch_size, -1)
                rpn_keep_ori = Variable(rpn_label_ori.view(-1).ne(-1).nonzero().view(-1))
                rpn_cls_score_ori = torch.index_select(rpn_cls_score_ori.view(-1, 2), 0, rpn_keep_ori)
                ################### distillation loss #################
                l1_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)  ###L1Loss
                # rcnn_cls_distil_loss=l1_loss_fn(cls_score_remove_add_cls,cls_score_org_rcnn) ### L2 loss
                ##### ce loss
                #cls_prob_org_rcnn = F.softmax(cls_score_org_rcnn, 1)
                rpn_cls_pred_ori = rpn_cls_score_ori.argmax(dim=1, keepdim=True).view(-1)
                self.rpn_cls_distil_loss = F.cross_entropy(rpn_cls_score, rpn_cls_pred_ori) ### cross_entropy
                
                #cls_preb_org_rcnn = F.softmax(cls_score_org_rcnn / T, 1)
                #rcnn_cls_distil_loss = alpha * SoftCrossEntropy(cls_score_remove_add_cls / T, cls_preb_org_rcnn,
                #                                                reduction='average')
                self.rpn_bbox_distil_loss = l1_loss_fn(rpn_bbox_pred_ori, rpn_bbox_pred)  ### l1 loss
                #######################################################################################
                '''
                ################################# distil rpn_conv loss ################################
                rpn_conv1_fea = rpn_conv1.squeeze(dim=0)  # .mul(base_feat.squeeze(dim=0))
                rpn_conv1_ori_fea = rpn_conv1_ori.squeeze(dim=0)  # .mul(base_feat_org.squeeze(dim=0))
                rpn_conv1_fea_att_sum_c = torch.mean(rpn_conv1_fea, dim=0)  # /base_feat.shape[1]
                rpn_conv1_ori_fea_att_sum_c = torch.mean(rpn_conv1_ori_fea, dim=0)  # /base_feat_org.shape[1]
                rpn_conv1_norm = torch.norm(rpn_conv1_fea_att_sum_c, p=2, keepdim=True)
                rpn_conv1_org_norm = torch.norm(rpn_conv1_ori_fea_att_sum_c, p=2, keepdim=True)
                # base_fea_norm = base_fea_att_sum_c/torch.norm(base_fea_att_sum_c, p=2, keepdim=True)
                # base_fea_org_norm = base_fea_org_att_sum_c/torch.norm(base_fea_org_att_sum_c, p=2, keepdim=True)
                l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
                self.rpn_conv1_distil_loss = l1_loss_fn(rpn_conv1_norm, rpn_conv1_org_norm)
                #######################################################################################
                return rois, self.rpn_loss_cls, self.rpn_loss_box,self.rpn_conv1_distil_loss#,self.rpn_cls_distil_loss,self.rpn_bbox_distil_loss

        return rois, self.rpn_loss_cls, self.rpn_loss_box


class _RPN_distil_residual(nn.Module):
    """ region proposal network """

    def __init__(self, din):
        super(_RPN_distil_residual, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2  # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4  # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, fasterRCNN_org=None, fasterRCNN_residual=None,base_feat_org=None,base_feat_residual=None,base_feat_inc=None):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        '''
        #################### cat old and new rpn pred #############################
        rpn_cls_prob=torch.cat((rpn_cls_prob_ori,rpn_cls_prob),dim=1)
        rpn_bbox_pred = torch.cat((rpn_bbox_pred_ori, rpn_bbox_pred), dim=1)
        rpn_cls_score = torch.cat((rpn_cls_score_ori,rpn_cls_score),dim=1)
        rpn_cls_score_reshape = torch.cat((rpn_cls_score_reshape_ori, rpn_cls_score_reshape), dim=1)
        ###########################################################################
        '''

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                  im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

            if fasterRCNN_org and fasterRCNN_residual:
                ################# ori fasterrcnn #####################################################
                rpn_conv1_ori = F.relu(fasterRCNN_org.RCNN_rpn.RPN_Conv(base_feat_org), inplace=True)
                rpn_conv1_residual = F.relu(fasterRCNN_residual.RCNN_rpn.RPN_Conv(base_feat_residual), inplace=True)
                rpn_conv1_inc = F.relu(self.RPN_Conv(base_feat_inc), inplace=True)
                '''
                rpn_cls_score_ori = fasterRCNN_org.RCNN_rpn.RPN_cls_score(rpn_conv1_ori)
                rpn_cls_score_reshape_ori = fasterRCNN_org.RCNN_rpn.reshape(rpn_cls_score_ori, 2)
                rpn_cls_prob_reshape_ori = F.softmax(rpn_cls_score_reshape_ori, 1)
                rpn_cls_prob_ori = fasterRCNN_org.RCNN_rpn.reshape(rpn_cls_prob_reshape_ori, fasterRCNN_org.RCNN_rpn.nc_score_out)
                rpn_bbox_pred_ori = fasterRCNN_org.RCNN_rpn.RPN_bbox_pred(rpn_conv1_ori)
                cfg_key_ori = 'TRAIN'

                rois_ori = fasterRCNN_org.RCNN_rpn.RPN_proposal((rpn_cls_prob_ori.data, rpn_bbox_pred_ori.data,
                                         im_info, cfg_key_ori))
                ############################################################################################
                '''
                '''
                ############################## distil rpn cls and bbox reg loss ##########################################
                rpn_data_ori = fasterRCNN_org.RCNN_rpn.RPN_anchor_target((rpn_cls_score_ori.data, gt_boxes, im_info, num_boxes))
                rpn_cls_score_ori = rpn_cls_score_reshape_ori.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
                rpn_label_ori = rpn_data_ori[0].view(batch_size, -1)
                rpn_keep_ori = Variable(rpn_label_ori.view(-1).ne(-1).nonzero().view(-1))
                rpn_cls_score_ori = torch.index_select(rpn_cls_score_ori.view(-1, 2), 0, rpn_keep_ori)
                ################### distillation loss #################
                l1_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)  ###L1Loss
                # rcnn_cls_distil_loss=l1_loss_fn(cls_score_remove_add_cls,cls_score_org_rcnn) ### L2 loss
                ##### ce loss
                #cls_prob_org_rcnn = F.softmax(cls_score_org_rcnn, 1)
                rpn_cls_pred_ori = rpn_cls_score_ori.argmax(dim=1, keepdim=True).view(-1)
                self.rpn_cls_distil_loss = F.cross_entropy(rpn_cls_score, rpn_cls_pred_ori) ### cross_entropy
    
                #cls_preb_org_rcnn = F.softmax(cls_score_org_rcnn / T, 1)
                #rcnn_cls_distil_loss = alpha * SoftCrossEntropy(cls_score_remove_add_cls / T, cls_preb_org_rcnn,
                #                                                reduction='average')
                self.rpn_bbox_distil_loss = l1_loss_fn(rpn_bbox_pred_ori, rpn_bbox_pred)  ### l1 loss
                #######################################################################################
                '''
                ################################# distil rpn_conv loss ################################
                rpn_conv1_fea = rpn_conv1_inc.squeeze(dim=0)#rpn_conv1  # .mul(base_feat.squeeze(dim=0))
                rpn_conv1_ori_fea = rpn_conv1_ori.squeeze(dim=0)  # .mul(base_feat_org.squeeze(dim=0))
                rpn_conv1_res_fea=rpn_conv1_fea-rpn_conv1_ori_fea
                rpn_conv1_residual_fea = rpn_conv1_residual.squeeze(dim=0)  # .mul(base_feat_org.squeeze(dim=0))
                #rpn_conv1_fea_att_sum_c = torch.mean(rpn_conv1_fea, dim=0)  # /base_feat.shape[1]
                #rpn_conv1_ori_fea_att_sum_c = torch.mean(rpn_conv1_ori_fea, dim=0)  # /base_feat_org.shape[1]
                rpn_conv1_res_fea_att_sum_c = torch.mean(rpn_conv1_res_fea, dim=0)  # /base_feat_org.shape[1]
                rpn_conv1_residual_fea_att_sum_c = torch.mean(rpn_conv1_residual_fea, dim=0)  # /base_feat_org.shape[1]

                #rpn_conv1_norm = torch.norm(rpn_conv1_fea_att_sum_c, p=2, keepdim=True)
                #rpn_conv1_org_norm = torch.norm(rpn_conv1_ori_fea_att_sum_c, p=2, keepdim=True)
                rpn_conv1_res_norm = torch.norm(rpn_conv1_res_fea_att_sum_c, p=2, keepdim=True)
                rpn_conv1_residual_norm = torch.norm(rpn_conv1_residual_fea_att_sum_c, p=2, keepdim=True)
                # base_fea_norm = base_fea_att_sum_c/torch.norm(base_fea_att_sum_c, p=2, keepdim=True)
                # base_fea_org_norm = base_fea_org_att_sum_c/torch.norm(base_fea_org_att_sum_c, p=2, keepdim=True)
                l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

                self.rpn_conv1_distil_loss = l1_loss_fn(rpn_conv1_res_norm, rpn_conv1_residual_norm)
                #######################################################################################
                return rois, self.rpn_loss_cls, self.rpn_loss_box, self.rpn_conv1_distil_loss  # ,self.rpn_cls_distil_loss,self.rpn_bbox_distil_loss
        return rois, self.rpn_loss_cls, self.rpn_loss_box