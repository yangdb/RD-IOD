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

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
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

    def forward(self, base_feat, im_info, gt_boxes, num_boxes,fasterRCNN_org):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)

        '''
        ############## distil ################################
        rpn_conv1_org = F.relu(fasterRCNN_org.RCNN_rpn.RPN_Conv(base_feat), inplace=True)
        
        rpn_conv1_att_sum_c = torch.mean(rpn_conv1.squeeze(), dim=0)  # /base_feat.shape[1]
        rpn_conv1_org_att_sum_c = torch.mean(rpn_conv1_org.squeeze(), dim=0)  # /base_feat_org.shape[1]
        rpn_conv1_norm = rpn_conv1_att_sum_c/torch.norm(rpn_conv1_att_sum_c, p=2, keepdim=True)
        rpn_conv1_org_norm = rpn_conv1_org_att_sum_c/torch.norm(rpn_conv1_org_att_sum_c, p=2, keepdim=True)
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        rpn_conv1_distil_loss = l1_loss_fn(rpn_conv1_norm, rpn_conv1_org_norm)
        
        #l1_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        #rpn_conv1_distil_loss = l1_loss_fn(rpn_conv1, rpn_conv1_org)
        ########################################################
        '''
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        ################## RPN embedding loss #################################
        l2_loss_fn_sum = torch.nn.MSELoss(reduce=True, size_average=True)  # , reduction = 'sum')
        l2_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        old_rpn_embed = fasterRCNN_org.RCNN_rpn.RPN_cls_score._parameters['weight'].data.view(18, -1) #torch.cat((fasterRCNN_org.RCNN_rpn.RPN_cls_score._parameters['weight'].data.view(18, -1),fasterRCNN_org.RCNN_rpn.RPN_cls_score._parameters['bias'].data.view(18, -1)), dim=1)
        old_fg = torch.mean(old_rpn_embed[::2], dim=0)
        old_bg = torch.mean(old_rpn_embed[1::2], dim=0)
        old_fg_norm = old_fg/torch.norm(old_fg, p=2, keepdim=True)#torch.norm(old_fg, p=2, keepdim=True)#
        old_bg_norm =old_bg/torch.norm(old_bg, p=2, keepdim=True) #torch.norm(old_bg, p=2, keepdim=True)#
        new_rpn_embed = self.RPN_cls_score._parameters['weight'].data.view(18, -1) #torch.cat((self.RPN_cls_score._parameters['weight'].data.view(18, -1),self.RPN_cls_score._parameters['bias'].data.view(18, -1)), dim=1)
        new_fg = torch.mean(new_rpn_embed[::2], dim=0)
        new_bg = torch.mean(new_rpn_embed[1::2], dim=0)
        new_fg_norm = new_fg/torch.norm(new_fg, p=2, keepdim=True)#torch.norm(new_fg, p=2, keepdim=True)#
        new_bg_norm = new_bg/torch.norm(new_bg, p=2, keepdim=True)#torch.norm(new_bg, p=2, keepdim=True)#
        # rpn_embed_distil_loss = l2_loss_fn_sum(new_rpn_embed,old_rpn_embed)/18
        # rpn_embed_distil_loss = l2_loss_fn_sum(new_fg,old_fg)+l2_loss_fn_sum(new_bg,old_bg)
        rpn_embed_distil_loss = l1_loss_fn(new_fg_norm, old_fg_norm) + l1_loss_fn(new_bg_norm, old_bg_norm)
        #######################################################################

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

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

        return rois, self.rpn_loss_cls, self.rpn_loss_box,rpn_embed_distil_loss#rpn_conv1_distil_loss
