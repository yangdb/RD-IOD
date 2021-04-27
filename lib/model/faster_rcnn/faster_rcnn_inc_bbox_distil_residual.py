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
#from model.rpn.rpn_dl import _RPN
#from model.rpn.rpn_distil import _RPN_distil as _RPN ####################### rpn distil
from model.rpn.rpn_distil import _RPN_distil_residual as _RPN ####################### rpn residual distil
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import ROIAlign, ROIPool
from model.roi_layers import nms
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
#from model.rpn.proposal_target_layer_cascade_distil import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

def SoftCrossEntropy(inputs, target, reduction='sum'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


class CosineSimilarity(nn.Module):
    """
    This similarity function simply computes the cosine similarity between each pair of vectors. It has
    no parameters.
    """
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

class L2Similarity(nn.Module):
    """
    This similarity function simply computes the l2 similarity between each pair of vectors. It has
    no parameters.
    """
    def forward(self, tensor_1, tensor_2):
        l2_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        l2_loss=0
        for t_1 in tensor_1:
            for t_2 in tensor_2:
                l2_loss+=l2_loss_fn(t_1,t_2)
        return l2_loss


def compute_iou(box1, box2, iou_thresh=0.3, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    box1 = box1[0:4]
    box2 = box2[0:4]

    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1])) # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6) #计算交并比
    if iou>iou_thresh:
        return True
    else:
        return False
    #return iou

class _fasterRCNN_inc_bbox_distil(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN_inc_bbox_distil, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_new_class =  cfg.NEW_CLASSES#10#5#1#5#5#10 #5 ############################# inc class num
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = torch.Tensor([0]).cuda()
        self.RCNN_loss_bbox = torch.Tensor([0]).cuda()

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, rois_org, cls_prob_org, bbox_pred_org, rois_label_org, fasterRCNN_org, step, roidb, ratio_index, fasterRCNN_residual):

        ########## frcnn_org_result #################
        scores = cls_prob_org.data
        boxes = rois_org.data[:, :, 1:5]
        batch_size=im_data.shape[0]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred_org.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(batch_size, -1, 4)#box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(batch_size, -1, 4 * (fasterRCNN_org.n_classes))#self.n_classes-1  box_deltas = box_deltas.view(1, -1, 4 * (fasterRCNN_org.n_classes))#self.n_classes-1

            pred_boxes = bbox_transform_inv(boxes, box_deltas, batch_size)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, batch_size)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        #pred_boxes /= im_info.data[0][2]

        scores = scores.squeeze(dim=0)
        pred_boxes = pred_boxes.squeeze(dim=0)

        thr_2=cfg.threshold_2
        if thr_2:
            thresh=0.9
        else:
            thresh=0.5#0.5#0.1#0.7#0.1#0.5#0.3#
        #print(scores.shape)
        org_det_gt_boxes=torch.Tensor().cuda()
        for j in range(1, scores.shape[1]):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS) #NMS=0.3
                cls_dets = cls_dets[keep.view(-1).long()]
                cls_tensor = torch.full([cls_dets.shape[0],1], j).cuda()
                cls_label_cat=torch.cat((cls_dets[:,0:4],cls_tensor),1)
                if org_det_gt_boxes.shape[0]==0:
                    org_det_gt_boxes=cls_label_cat
                else:
                    org_det_gt_boxes=torch.cat((org_det_gt_boxes,cls_label_cat), 0)

        #############################################

        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        ####compute IOU between gt_boxes and rpn_org_proposals, delete overlapped bboxes of rpn_org_proposals
        final_org_det=torch.Tensor().cuda()
        for o_bbox in org_det_gt_boxes:
            uq_flag=True
            for gt_bbox in gt_boxes.squeeze()[:num_boxes]:
                if compute_iou(o_bbox,gt_bbox): #iou=0.3
                    uq_flag=False
                    break
            if uq_flag:
                if final_org_det.shape[0]==0:
                    final_org_det=o_bbox.unsqueeze(dim=0)
                else:
                    final_org_det=torch.cat((final_org_det,o_bbox.unsqueeze(dim=0)),0)
        org_det_gt_boxes=final_org_det#.unsqueeze(dim=0)
        ###################################################
        gt_boxes_n=gt_boxes.clone()
        num_boxes_n=num_boxes.clone()
        gt_boxes=torch.cat((org_det_gt_boxes,gt_boxes.squeeze()),0).unsqueeze(dim=0)  ### not use pesudo
        num_boxes+=org_det_gt_boxes.shape[0] ### not use pesudo
        ###########################################################

        if thr_2:
            ########################## 2-threshold training ########################
            thresh_lower = 0.1  # 0.9#0.9#0.1#0.7#0.1#0.5#0.3#
            # print(scores.shape)
            org_det_gt_boxes_lower = torch.Tensor().cuda()
            for j in range(1, scores.shape[1]):
                inds = torch.nonzero(scores[:, j] > thresh_lower).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)  # NMS=0.3
                    cls_dets = cls_dets[keep.view(-1).long()]
                    cls_tensor = torch.full([cls_dets.shape[0], 1], j).cuda()
                    cls_label_cat = torch.cat((cls_dets[:, 0:4], cls_tensor), 1)
                    if org_det_gt_boxes_lower.shape[0] == 0:
                        org_det_gt_boxes_lower = cls_label_cat
                    else:
                        org_det_gt_boxes_lower = torch.cat((org_det_gt_boxes_lower, cls_label_cat), 0)
            gt_boxes_n1 = gt_boxes_n.clone()
            num_boxes_n1 = num_boxes_n.clone()
            gt_boxes_lower = gt_boxes_n1.data
            num_boxes_lower = num_boxes_n1.data
            final_org_det = torch.Tensor().cuda()
            for o_bbox in org_det_gt_boxes_lower:
                uq_flag = True
                for gt_bbox in gt_boxes_lower.squeeze()[:num_boxes_lower]:
                    if compute_iou(o_bbox, gt_bbox):  # iou=0.3
                        uq_flag = False
                        break
                if uq_flag:
                    if final_org_det.shape[0] == 0:
                        final_org_det = o_bbox.unsqueeze(dim=0)
                    else:
                        final_org_det = torch.cat((final_org_det, o_bbox.unsqueeze(dim=0)), 0)
            org_det_gt_boxes_lower = final_org_det  # .unsqueeze(dim=0)
            gt_boxes_lower = torch.cat((org_det_gt_boxes_lower, gt_boxes_lower.squeeze()), 0).unsqueeze(dim=0)
            num_boxes_lower += org_det_gt_boxes_lower.shape[0]
            ########################################################################



        ###################################################################################################################
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        base_feat_res = fasterRCNN_residual.RCNN_base(im_data)
        ##################### feature distil ###############################
        base_feat_org = fasterRCNN_org.RCNN_base(im_data)

        l2_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        #base_feat_distil_loss = l2_loss_fn(base_feat, base_feat_org)


        base_fea_2 = base_feat.squeeze(dim=0)#.mul(base_feat.squeeze(dim=0))
        base_fea_org_2 = base_feat_org.squeeze(dim=0)#.mul(base_feat_org.squeeze(dim=0))
        base_fea_att_sum_c = torch.mean(base_fea_2, dim=0) #/base_feat.shape[1]
        base_fea_org_att_sum_c = torch.mean(base_fea_org_2, dim=0) #/base_feat_org.shape[1]
        base_fea_norm = torch.norm(base_fea_att_sum_c, p=2, keepdim=True)
        base_fea_org_norm = torch.norm(base_fea_org_att_sum_c, p=2, keepdim=True)
        #base_fea_norm = base_fea_att_sum_c/torch.norm(base_fea_att_sum_c, p=2, keepdim=True)
        #base_fea_org_norm = base_fea_org_att_sum_c/torch.norm(base_fea_org_att_sum_c, p=2, keepdim=True)
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        base_feat_distil_loss = l1_loss_fn(base_fea_norm, base_fea_org_norm)#(base_feat,base_feat_org)# ablation (base_feat,base_feat_org)#
        #base_feat_distil_loss = l1_loss_fn(base_feat,base_feat_org)




        base_feat_residual=base_feat - base_feat_org
        base_feat_residual_sq = base_feat_residual.squeeze(dim=0)  # .mul(base_feat.squeeze(dim=0))
        base_feat_res_sq = base_feat_res.squeeze(dim=0)  # .mul(base_feat_org.squeeze(dim=0))
        base_feat_residual_sum_c = torch.mean(base_feat_residual_sq, dim=0)  # /base_feat.shape[1]
        base_feat_res_sum_c = torch.mean(base_feat_res_sq, dim=0)  # /base_feat_org.shape[1]
        base_feat_residual_norm = torch.norm(base_feat_residual_sum_c, p=2, keepdim=True)
        base_feat_res_norm = torch.norm(base_feat_res_sum_c, p=2, keepdim=True)

        #base_feat_residual_norm_div=base_feat_residual_sum_c/10*base_feat_residual_norm
        #base_feat_res_norm_div=base_feat_res_sum_c/10*base_feat_res_norm
        base_feat_residual_loss = l1_loss_fn(base_feat_residual_norm,base_feat_res_norm)#(base_feat_residual_norm_div,base_feat_res_norm_div)#(base_feat_residual_sum_c,base_feat_res_sum_c)#(base_feat_residual_norm,base_feat_res_norm)#(base_feat_residual,base_feat_res)#(base_feat_residual_norm,base_feat_res_norm)
        #base_feat_residual_loss = l1_loss_fn(base_feat_residual,     base_feat_res)



        #base_feat_residual_loss = l1_loss_fn(base_feat - base_feat_res, base_feat_org) + l1_loss_fn(
        #    base_feat - base_feat_org, base_feat_res)

        #base_feat_distil_loss_new = l1_loss_fn(base_fea_norm,base_feat_res_norm)
        #base_feat_distil_loss+=base_feat_distil_loss_new
        #base_feat_residual_loss_inc= l1_loss_fn(torch.norm(torch.mean(((base_feat+(base_feat_org+base_feat_res))/2).squeeze(dim=0), dim=0), p=2, keepdim=True), base_fea_norm)
        #base_feat_residual_loss_inc= l1_loss_fn(torch.norm(torch.mean((base_feat_org+base_feat_res).squeeze(dim=0), dim=0), p=2, keepdim=True), base_fea_norm)
        base_feat_residual_loss_inc= l1_loss_fn(torch.norm(torch.mean((base_feat_org+base_feat_res).squeeze(dim=0), dim=0), p=2, keepdim=True), base_fea_norm)
        #base_feat_residual_loss_inc = l1_loss_fn(    base_feat_org + base_feat_res,        base_feat)

        base_feat_residual_loss=base_feat_residual_loss_inc+base_feat_residual_loss

        #base_feat_add=(base_feat+(base_feat_org+base_feat_res))/2#base_feat#base_feat_org+base_feat_res ### + * base_feat+base_feat_res
        base_feat_add=base_feat

        #####################################################################

        # feed base feature map tp RPN to obtain rois
        #rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        if thr_2:
            rois_lower, rpn_loss_cls_lower, rpn_loss_bbox_lower= self.RCNN_rpn(base_feat_add, im_info, gt_boxes_lower, num_boxes_lower) #### rpn original
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_add, im_info, gt_boxes, num_boxes)  #### rpn original
        #rois, rpn_loss_cls, rpn_loss_bbox,rpn_conv1_distil_loss= self.RCNN_rpn(base_feat_add, im_info, gt_boxes, num_boxes,fasterRCNN_org)### rpn conv optional
        #rois, rpn_loss_cls, rpn_loss_bbox,rpn_conv1_distil_loss= self.RCNN_rpn(base_feat_add, im_info, gt_boxes, num_boxes,\
        #             fasterRCNN_org,fasterRCNN_residual,base_feat_org=base_feat_org,base_feat_residual=base_feat_res,base_feat_inc=base_feat)### rpn residual conv optional

        gt_boxes_n[0, gt_boxes_n.squeeze()[:, 4].nonzero().view(-1).long(), 4] = \
            gt_boxes_n[0, gt_boxes_n.squeeze()[:, 4].nonzero().view(-1).long(), 4] - (self.n_classes - self.n_new_class - 1)
        rois_r, rpn_loss_cls_r, rpn_loss_bbox_r = fasterRCNN_residual.RCNN_rpn(base_feat_res, im_info, gt_boxes_n, num_boxes_n)

        #rois_r_c, rpn_loss_cls_r_c, rpn_loss_bbox_r_c = fasterRCNN_residual.RCNN_rpn(base_feat_res_c, im_info, gt_boxes_n,
        #                                                                      num_boxes_n)
        #rpn_loss_cls_r+=rpn_loss_cls_r_c
        #rpn_loss_bbox_r+=rpn_loss_bbox_r_c



        #rois, rpn_loss_cls, rpn_loss_bbox, \
        #    =self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, fasterRCNN_org=fasterRCNN_org) ########### rpn_distil
        #rpn_cls_distil_loss, rpn_bbox_distil_loss

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            roi_data_r = fasterRCNN_residual.RCNN_proposal_target(rois_r, gt_boxes_n, num_boxes_n)
            rois_r, rois_label_r, rois_target_r, rois_inside_ws_r, rois_outside_ws_r = roi_data_r
            rois_label_r = Variable(rois_label_r.view(-1).long())
            rois_target_r = Variable(rois_target_r.view(-1, rois_target_r.size(2)))
            rois_inside_ws_r = Variable(rois_inside_ws_r.view(-1, rois_inside_ws_r.size(2)))
            rois_outside_ws_r = Variable(rois_outside_ws_r.view(-1, rois_outside_ws_r.size(2)))
        else:
            print(gt_boxes.shape)
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = torch.Tensor([0]).cuda()
            rpn_loss_bbox = torch.Tensor([0]).cuda()

            rois_label_r = None
            rois_target_r = None
            rois_inside_ws_r = None
            rois_outside_ws_r = None
            rpn_loss_cls_r = torch.Tensor([0]).cuda()
            rpn_loss_bbox_r = torch.Tensor([0]).cuda()



        rois = Variable(rois)
        rois_r = Variable(rois_r)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat_add, rois.view(-1, 5))
            #pooled_feat_inc = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat_org = fasterRCNN_org.RCNN_roi_align(base_feat_org, rois.view(-1, 5))#self
            pooled_feat_r = fasterRCNN_residual.RCNN_roi_align(base_feat_res, rois_r.view(-1, 5))
            pooled_feat_r_roiinc=fasterRCNN_residual.RCNN_roi_align(base_feat_res, rois.view(-1, 5))



        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat_add, rois.view(-1,5))
            #pooled_feat_inc = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
            pooled_feat_org = fasterRCNN_org.RCNN_roi_pool(base_feat_org, rois.view(-1, 5))#self
            pooled_feat_r = fasterRCNN_residual.RCNN_roi_pool(base_feat_res, rois_r.view(-1, 5))
            pooled_feat_r_roiinc = fasterRCNN_residual.RCNN_roi_pool(base_feat_res, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        #pooled_feat_inc = self._head_to_tail(pooled_feat_inc)
        pooled_feat_org = fasterRCNN_org._head_to_tail(pooled_feat_org)#self
        pooled_feat_r = fasterRCNN_residual._head_to_tail(pooled_feat_r)
        pooled_feat_r_roiinc =fasterRCNN_residual._head_to_tail(pooled_feat_r_roiinc)

        #pooled_feat_distil_loss = l1_loss_fn(torch.norm(torch.mean((pooled_feat - pooled_feat_org).squeeze(dim=0), dim=0), p=2, keepdim=True), torch.norm(torch.mean(pooled_feat_r_roiinc.squeeze(dim=0), dim=0), p=2, keepdim=True))
        pooled_feat_distil_loss =  l1_loss_fn(pooled_feat-pooled_feat_org, pooled_feat_r_roiinc)\
            +l1_loss_fn(pooled_feat, pooled_feat_org+pooled_feat_r_roiinc)#+l1_loss_fn(pooled_feat-pooled_feat_r_roiinc, pooled_feat_org)
                                  #l1_loss_fn(pooled_feat-pooled_feat_org, pooled_feat_r_roiinc) \
            #l1_loss_fn(pooled_feat-pooled_feat_r_roiinc, pooled_feat_org)+ l1_loss_fn(pooled_feat-pooled_feat_org, pooled_feat_r_roiinc)
        #pooled_feat_distil_loss =pooled_feat_distil_loss_1


        ##########################################################################


        ############ compute org_rcnn bbox_pred and cls_score ####################
        bbox_pred_org_rcnn=fasterRCNN_org.RCNN_bbox_pred(pooled_feat)
        cls_score_org_rcnn=fasterRCNN_org.RCNN_cls_score(pooled_feat)
        #if self.training and not self.class_agnostic:
        ##########################################################################

        # compute bbox offset
        bbox_pred_old = self.RCNN_bbox_pred(pooled_feat)
        bbox_pred_residual = fasterRCNN_residual.RCNN_bbox_pred(pooled_feat_r)
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

            bbox_pred_view_r = bbox_pred_residual.view(bbox_pred_residual.size(0), int(bbox_pred_residual.size(1) / 4), 4)
            bbox_pred_select_r = torch.gather(bbox_pred_view_r, 1, rois_label_r.view(rois_label_r.size(0), 1, 1).expand(rois_label_r.size(0), 1, 4))
            bbox_pred_r = bbox_pred_select_r.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        cls_score_r = fasterRCNN_residual.RCNN_cls_score(pooled_feat_r)
        cls_prob_r = F.softmax(cls_score_r,1)#torch.cat((cls_score_r[0],cls_score_r[-self.n_new_class:]),dim=0), 1)

        cls_score_r_roiinc = fasterRCNN_residual.RCNN_cls_score(pooled_feat_r_roiinc)


        ################# split score fc (old and new)#####################
        cls_score_new = self.RCNN_cls_score_new(pooled_feat)
        cls_score_cat = torch.cat((cls_score,cls_score_new),dim=1)
        cls_prob = F.softmax(cls_score_cat,1)
        ###################################################################





        RCNN_loss_cls = torch.Tensor([0]).cuda()
        RCNN_loss_bbox = torch.Tensor([0]).cuda()

        if self.training:

            #alpha = 1/20
            # classification loss
            #RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_cls = F.cross_entropy(cls_score_cat,rois_label)################## split old and new
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            RCNN_loss_cls_r = F.cross_entropy(cls_score_r, rois_label_r)  ################## split old and new
            # bounding box regression L1 loss
            RCNN_loss_bbox_r = _smooth_l1_loss(bbox_pred_r, rois_target_r, rois_inside_ws_r, rois_outside_ws_r)

            #rcnn_cls_distil_loss=0
            ################### distillation loss #################
            l1_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
            cls_score_remove_add_cls=cls_score#[:,:-1]  # split old and new


            old_cls_score=cls_score[:,:] #cls_score[:,1:]#
            old_cls_score_org=cls_score_org_rcnn[:,:]#cls_score_org_rcnn[:,1:]#
            old_cls_score_softmax = F.softmax(old_cls_score, 1)
            old_cls_score_org_softmax = F.softmax(old_cls_score_org, 1)
            rcnn_cls_distil_loss = l1_loss_fn(old_cls_score_softmax, old_cls_score_org_softmax)

            new_cls_score = cls_score_new[:, :]
            new_cls_score_residual = cls_score_r_roiinc[:,1:]#cls_score_r[:, 1:]
            new_cls_score_softmax = F.softmax(new_cls_score, 1)
            new_cls_score_residual_softmax = F.softmax(new_cls_score_residual, 1)
            rcnn_cls_distil_loss_r = l1_loss_fn(new_cls_score_softmax, new_cls_score_residual_softmax)

            #rcnn_cls_distil_loss = l1_loss_fn(cls_score[:,1:], cls_score_org_rcnn[:,1:])
            #rcnn_cls_distil_loss_r = l1_loss_fn(cls_score_new[:,:], cls_score_r[:,1:])

            rcnn_cls_distil_loss+=rcnn_cls_distil_loss_r

            ################### bbox distillation loss ############
            bbox_pred_residual_roiinc = fasterRCNN_residual.RCNN_bbox_pred(pooled_feat_r_roiinc)
            rcnn_bbox_distil_loss = l1_loss_fn(bbox_pred_old,bbox_pred_org_rcnn)### l1 loss[:,4:]
            #rcnn_bbox_distil_loss = l1_loss(bbox_pred_old, bbox_pred_org_rcnn)
            rcnn_bbox_distil_loss+=l1_loss_fn(bbox_pred_new,bbox_pred_residual_roiinc[:,4:])
            #######################################################



        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        cls_prob_r = cls_prob_r.view(batch_size, rois_r.size(1), -1)
        bbox_pred_r = bbox_pred_r.view(batch_size, rois_r.size(1), -1)


        ##### ablation ##############
        #rcnn_cls_distil_loss=torch.Tensor([0]).cuda()
        #rcnn_cls_distil_loss=RCNN_loss_cls_s
        rcnn_bbox_distil_loss=torch.Tensor([0]).cuda()
        #base_feat_distil_loss=torch.Tensor([0]).cuda()
        #RCNN_loss_cls=RCNN_loss_cls+RCNN_loss_cls_new
        rpn_conv1_distil_loss = torch.Tensor([0]).cuda()
        #pooled_feat_distil_loss+=pooled_feat_distil_loss_1
        #pooled_feat_distil_loss = torch.Tensor([0]).cuda()
        cos_loss=torch.Tensor([0]).cuda()
        #base_feat_residual_loss=torch.Tensor([0]).cuda()


        if thr_2:
            return rois, cls_prob, bbox_pred, rpn_loss_cls_lower, rpn_loss_bbox_lower, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
               rcnn_cls_distil_loss, rcnn_bbox_distil_loss, base_feat_distil_loss, rpn_conv1_distil_loss, pooled_feat_distil_loss, cos_loss, \
               rpn_loss_cls_r, rpn_loss_bbox_r, RCNN_loss_cls_r, RCNN_loss_bbox_r, \
               base_feat_residual_loss #rpn_loss_cls_lower, rpn_loss_bbox_lower
        else:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
               rcnn_cls_distil_loss, rcnn_bbox_distil_loss, base_feat_distil_loss, rpn_conv1_distil_loss, pooled_feat_distil_loss, cos_loss, \
               rpn_loss_cls_r, rpn_loss_bbox_r, RCNN_loss_cls_r, RCNN_loss_bbox_r, \
               base_feat_residual_loss
        # ,rpn_embed_distil_loss
        #,rpn_cls_distil_loss,rpn_bbox_distil_loss #,margin_loss


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
