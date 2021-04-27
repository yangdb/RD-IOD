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
def correlation_distillation_loss(fea1,fea2,fea1_old,fea2_old):
    fea1 = fea1.reshape(fea1.shape[0], fea1.shape[1], -1)
    fea2 = fea2.reshape(fea2.shape[0], fea2.shape[1], -1)
    fea1_old = fea1_old.reshape(fea1_old.shape[0], fea1_old.shape[1], -1)
    fea2_old = fea2_old.reshape(fea2_old.shape[0], fea2_old.shape[1], -1)
    corr_loss = torch.Tensor([0]).cuda()
    for i in range(0, fea1.shape[0]):
        fea1_norm = F.normalize(fea1[i], dim=1)### dim=0 20210222ydb
        fea2_norm = F.normalize(fea2[i], dim=1)### dim=0
        fea1_old_norm = F.normalize(fea1_old[i], dim=1)### dim=0
        fea2_old_norm = F.normalize(fea2_old[i], dim=1)### dim=0
        sim_matrix_org = fea2_old_norm.mm(fea1_old_norm.t())
        sim_matrix = fea2_norm.mm(fea1_norm.t())
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        corr_loss += l1_loss_fn(sim_matrix, sim_matrix_org)
    return corr_loss


def correlation_distillation_cosloss(fea1,fea2,fea1_old,fea2_old):
    fea1 = fea1.reshape(fea1.shape[0], fea1.shape[1], -1)
    fea2 = fea2.reshape(fea2.shape[0], fea2.shape[1], -1)
    fea1_old = fea1_old.reshape(fea1_old.shape[0], fea1_old.shape[1], -1)
    fea2_old = fea2_old.reshape(fea2_old.shape[0], fea2_old.shape[1], -1)
    sim_matrix_org = torch.zeros(fea1.shape[0],fea2.shape[1], fea1.shape[1]).cuda()
    sim_matrix = torch.zeros(fea1.shape[0],fea2.shape[1], fea1.shape[1]).cuda()
    corr_loss=torch.Tensor([0]).cuda()
    for i in range(0, fea1.shape[0]):
        fea1_norm = F.normalize(fea1[i], dim=1)### dim=0 20210222ydb
        fea2_norm = F.normalize(fea2[i], dim=1)### dim=0
        fea1_old_norm = F.normalize(fea1_old[i], dim=1)### dim=0
        fea2_old_norm = F.normalize(fea2_old[i], dim=1)### dim=0
        sim_matrix_org[i] = fea2_old_norm.mm(fea1_old_norm.t())
        sim_matrix[i] = fea2_norm.mm(fea1_norm.t())
        corr_loss += 0.00001*torch.norm((sim_matrix_org[i]-sim_matrix[i]).mul(sim_matrix_org[i]-sim_matrix[i]))
    corr_loss=corr_loss/fea1.shape[0]
    return corr_loss



class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x),y

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

        self.n_new_classes=cfg.NEW_CLASSES

        # self.channel_att = SE_Block(512)
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

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, fasterRCNN_org=None, fasterRCNN_residual=None,base_feat_org=None,base_feat_residual=None,base_feat_inc=None,reproduce_flag=''):
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

        batch_size = base_feat.size(0)
        ######## 从old中采样前景位置，从new中采样背景位置，前景与背景特征计算相似度，两个模型拟合该相似度
        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # rpn_conv1,c_att = self.channel_att(rpn_conv1)
        # get rpn classification score
        if base_feat_org is not None:
            rpn_conv1_org = F.relu(fasterRCNN_org.RCNN_rpn.RPN_Conv(base_feat_org), inplace=True)#[2,512,38,50]
            # rpn_conv1_org,c_att_org = fasterRCNN_org.RCNN_rpn.channel_att(rpn_conv1_org)
        # if base_feat_org is not None:
        #     rpn_conv1_org = F.relu(fasterRCNN_org.RCNN_rpn.RPN_Conv(base_feat_org), inplace=True)#[2,512,38,50]
        #     rpn_conv1_org,c_att_org = fasterRCNN_org.RCNN_rpn.channel_att(rpn_conv1_org)
        #
        #     th_dict={'1':0.1+1/20/2,'5':5/20/2,'10':10/20}
        #     low_th=5/20
        #     high_th=1-5/20
        #     low_th=0.5#0.5 #0.4
        #     high_th=0.5#+(self.n_new_classes/20)#0.5#0.5#0.6
        #     max_a = max(c_att_org.view(-1))
        #     min_a = min(c_att_org.view(-1))
        #     channel_att_norm = (((c_att_org.view(-1)) - min_a) / (max_a - min_a + 0.000001))
        #     channel_att_norm_new = (((c_att.view(-1)) - min(c_att.view(-1))) / (max(c_att.view(-1))- min(c_att.view(-1)) + 0.000001))
        #     c_att_idx_low = (channel_att_norm < low_th).nonzero().view(-1)
        #     c_att_idx_high = (channel_att_norm >high_th).nonzero().view(-1)
        #     # # print(self.n_new_classes,int(c_att_org.shape[1]/2 * ((20-self.n_new_classes) / 20)),channel_att_norm.shape)
        #     topk= int(c_att_org.shape[1]/2)#int(c_att_org.shape[1]/2 * ((20-self.n_new_classes) / 20)
        #     high_v, high_top = torch.topk(channel_att_norm,topk,sorted=True)
        #     low_v, low_top = torch.topk(channel_att_norm, topk, sorted=True, largest=False)
        #     high_v_new, high_top_new = torch.topk(channel_att_norm_new, topk, sorted=True)
        #     # c_att_idx_low = low_top#[low_v<0.5]
        #     # c_att_idx_high = high_top#[high_v>0.5]
        #     # c_att_idx_high_new = high_top_new
        #     # # print(c_att_idx_high.shape,c_att_idx_low.shape)
        #
        #
        #     base_feat_high = rpn_conv1[:,c_att_idx_high.view(-1).long()]#c_att_idx_high
        #     base_feat_high = base_feat_high.reshape(base_feat_high.shape[0],base_feat_high.shape[1],-1)
        #     base_feat_org_high = rpn_conv1_org[:,c_att_idx_high.view(-1).long()]#c_att_idx_high
        #     base_feat_org_high = base_feat_org_high.reshape(base_feat_org_high.shape[0], base_feat_org_high.shape[1], -1)
        #     base_feat_low = rpn_conv1[:, c_att_idx_low.view(-1).long()]#c_att_idx_low
        #     base_feat_low = base_feat_low.reshape(base_feat_low.shape[0], base_feat_low.shape[1], -1)
        #     base_feat_org_low = rpn_conv1_org[:, c_att_idx_low.view(-1).long()]#c_att_idx_low
        #     base_feat_org_low = base_feat_org_low.reshape(base_feat_org_low.shape[0], base_feat_org_low.shape[1], -1)
        #     # base_feat_distil_loss = correlation_distillation_loss(base_feat_low,base_feat_high,base_feat_org_low,base_feat_org_high)
        #     base_feat_distil_loss = correlation_distillation_loss(base_feat_high, base_feat_high, base_feat_org_high, base_feat_org_high)#+ correlation_distillation_loss(base_feat_low, base_feat_low, base_feat_org_low,
        #     # base_feat_distil_loss = l1_loss_fn(torch.norm(torch.mean(base_feat_high.squeeze(dim=0),dim=0), p=2, keepdim=True),torch.norm(torch.mean(base_feat_org_high.squeeze(dim=0),dim=0), p=2, keepdim=True))
        #     # base_feat_distil_loss = l1_loss_fn(base_feat_high, base_feat_org_high)
        #     # base_feat_distil_loss += l1_loss_fn(c_att, c_att_org)
        #     #                                                      # base_feat_org_low)
        #     # #                      + l1_loss_fn(c_att_org.view(-1), c_att.view(-1))
        #     # #base_feat_distil_loss = l1_loss_fn(rpn_conv1*channel_att_norm.view(rpn_conv1.shape[0], rpn_conv1.shape[1], 1, 1).expand_as(rpn_conv1),rpn_conv1_org*channel_att_norm.view(rpn_conv1.shape[0], rpn_conv1.shape[1], 1, 1).expand_as(rpn_conv1))
        #
        #     # max_a_new = max(c_att.view(-1))
        #     # min_a_new = min(c_att.view(-1))
        #     # channel_att_norm_new = (((c_att.view(-1)) - min_a_new) / (max_a_new - min_a_new + 0.000001))
        #     # log_a = F.log_softmax(channel_att_norm_new)
        #     # softmax_b = F.softmax(channel_att_norm, dim=-1)
        #     # base_feat_distil_loss = base_feat_distil_loss#+F.kl_div(log_a,softmax_b)#
        #
        #
        #
        #     att=torch.mean(torch.abs(rpn_conv1_org),dim=1)#,_ max
        #     att=att.view(rpn_conv1.shape[0],-1)
        #     min_v,_=torch.min(att,dim=1)
        #     max_v, _ = torch.max(att, dim=1)
        #
        #     att_norm = ((att - min_v.unsqueeze(dim=1).repeat(1,att.shape[1])) / (max_v - min_v + 0.000001).unsqueeze(dim=1).repeat(1,att.shape[1]))#.view(-1)#.view(rpn_conv1_org.shape[0],rpn_conv1_org.shape[2],rpn_conv1_org.shape[3]) ## transfer the high response points relation to low response points
        #     # if att_norm.shape[1]==2166:
        #     #     print(att_norm)
        #     import random
        #
        #     high_point=0.8#0.8
        #     low_point=0.1#0.1
        #     high_idx=(att_norm > high_point).nonzero()#.view(-1)
        #
        #     # high_idx_0=high_idx[random.sample(range(0, len(high_idx[high_idx[:]==0])), 50)]
        #     # high_idx_1 = high_idx[random.sample(range(0, len(high_idx[high_idx[:] == 1])), 50)]
        #     # high_idx=torch.cat(high_idx_0,high_idx_1)
        #
        #
        #     #print(att_norm)
        #     # print((att_norm < 0.1))
        #     low_idx = (att_norm < low_point).nonzero()  # (att_norm_new < 0.1).nonzero()#.view(-1)
        #     # high_idx = high_idx[
        #     #     random.sample(range(0, len(high_idx[high_idx[:] == 0])), min(50, len(high_idx[high_idx[:] == 0])))]
        #
        #     att_norm_org_high = att_norm[high_idx[:, 0].long(), high_idx[:, 1].long()]  # [high_idx.long()]
        #     att_new = torch.mean(torch.abs(rpn_conv1), dim=1)#, _ max
        #     att_new = att_new.view(rpn_conv1.shape[0], -1)
        #     min_v_new, _ = torch.min(att_new, dim=1)
        #     max_v_new, _ = torch.max(att_new, dim=1)
        #     att_norm_new = ((att_new - min_v_new.unsqueeze(dim=1).repeat(1,att_new.shape[1])) / (max_v_new - min_v_new + 0.000001).unsqueeze(dim=1).repeat(1,att_new.shape[1]))#.view(-1)  # .view(rpn_conv1_org.shape[0],rpn_conv1_org.shape[2],rpn_conv1_org.shape[3]) ## transfer the high response points relation to low response points
        #
        #
        #
        #     # low_idx = low_idx[random.sample(range(0, len(low_idx)), 300)]
        #     att_norm_new_high = att_norm_new[high_idx[:,0].long(),high_idx[:,1].long()]#[high_idx.long()]
        #
        #     l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        #     att_loss=l1_loss_fn(att_norm_org_high,att_norm_new_high)
        #     #att_norm = torch.uint8(255 * att_norm)
        #
        #
        #     high_org = rpn_conv1_org.view(rpn_conv1_org.shape[0], rpn_conv1_org.shape[1], -1)[:, :, high_idx[:,1].long()]#[high_idx[:,0].long(), :, high_idx[:,1].long()]#[:, :, high_idx.long()]
        #     low_org = rpn_conv1_org.view(rpn_conv1_org.shape[0], rpn_conv1_org.shape[1], -1)[:, :, low_idx[:,1].long()]#[low_idx[:,0].long(), :, low_idx[:,1].long()]#[:, :, low_idx.long()]
        #     high_new = rpn_conv1.view(rpn_conv1.shape[0], rpn_conv1.shape[1], -1)[:, :, high_idx[:,1].long()]#[high_idx[:,0].long(), :, high_idx[:,1].long()]#[:, :, high_idx.long()]
        #     low_new = rpn_conv1.view(rpn_conv1.shape[0], rpn_conv1.shape[1], -1)[:, :, low_idx[:,1].long()]#[low_idx[:,0].long(), :, low_idx[:,1].long()]#[:, :, low_idx.long()]
        #     high_org_norm = F.normalize(high_org, dim=1)
        #     low_org_norm = F.normalize(low_org, dim=1)
        #     high_new_norm = F.normalize(high_new, dim=1)
        #     low_new_norm = F.normalize(low_new, dim=1)
        #
        #     sim_matrix_org = torch.zeros(high_org_norm.shape[0], high_org_norm.shape[2], low_org_norm.shape[2]).cuda()
        #     sim_matrix_new = torch.zeros(high_new_norm.shape[0], high_new_norm.shape[2], low_new_norm.shape[2]).cuda()
        #     # if torch.cuda.is_available():
        #     #     sim_matrix_new=sim_matrix_new.cuda()
        #     #     sim_matrix_org=sim_matrix_org.cuda()
        #     for b in range(high_org_norm.shape[0]):
        #         sim_matrix_org[b] = high_org_norm[b].t().mm(low_org_norm[b])
        #     for b in range(high_new_norm.shape[0]):
        #         sim_matrix_new[b] = high_new_norm[b].t().mm(low_new_norm[b])
        #     corr_loss = torch.norm((sim_matrix_org-sim_matrix_new).mul(sim_matrix_org-sim_matrix_new))
        #     #att_loss = torch.Tensor([0]).cuda()
        #     #corr_loss = torch.Tensor([0]).cuda()
        #
        #     # corr_loss = l1_loss_fn(high_new,high_org)+l1_loss_fn(low_new,low_org)
        # else:
        #     base_feat_distil_loss = torch.Tensor([0]).cuda()
        #     corr_loss = torch.Tensor([0]).cuda()

        rpn_cls_score = self.RPN_cls_score(rpn_conv1)#[2,18,38,50]
        if fasterRCNN_org:
            rpn_cls_score_org = fasterRCNN_org.RCNN_rpn.RPN_cls_score(rpn_conv1_org)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)#[2,2,342,50]
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        if fasterRCNN_org:
            rpn_cls_score_reshape_org = fasterRCNN_org.RCNN_rpn.reshape(rpn_cls_score_org, 2)
            rpn_cls_prob_reshape_org = F.softmax(rpn_cls_score_reshape_org, 1)  # [2,2,342,50]
            rpn_cls_prob_org = fasterRCNN_org.RCNN_rpn.reshape(rpn_cls_prob_reshape_org, fasterRCNN_org.RCNN_rpn.nc_score_out)
            rpn_bbox_pred_org = fasterRCNN_org.RCNN_rpn.RPN_bbox_pred(rpn_conv1_org)


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


        if reproduce_flag=='iccv17':
            base_feat_distil_loss = torch.Tensor([0]).cuda()
            corr_loss = torch.Tensor([0]).cuda()
        elif reproduce_flag=='icme':
            base_feat_distil_loss = F.mse_loss(rpn_conv1, rpn_conv1_org)
        elif reproduce_flag=='ijcnn':
            rpn_dsl_loss= F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_pred_org) + F.smooth_l1_loss(rpn_cls_score, rpn_cls_score_org)
            base_feat_distil_loss = rpn_dsl_loss
        else:
            base_feat_distil_loss = torch.Tensor([0]).cuda()
            corr_loss = torch.Tensor([0]).cuda()

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


        if not self.training:
            corr_loss = torch.Tensor([0]).cuda()

        corr_loss = torch.Tensor([0]).cuda()
        # base_feat_distil_loss = torch.Tensor([0]).cuda()
        # base_feat_distil_loss = l1_loss_fn(rpn_conv1, rpn_conv1_org)

        return rois, self.rpn_loss_cls, self.rpn_loss_box,base_feat_distil_loss+corr_loss#corr_loss#base_feat_distil_loss+ corr_loss#