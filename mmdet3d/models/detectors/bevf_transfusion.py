import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .bevf_faster_rcnn import BEVF_FasterRCNN


@DETECTORS.register_module()
class BEVF_TransFusion(BEVF_FasterRCNN):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(BEVF_TransFusion, self).__init__(**kwargs)

        self.freeze_img = kwargs.get('freeze_img', False)
        self.init_weights(pretrained=kwargs.get('pretrained', None))

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(BEVF_TransFusion, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        #points = [torch.zeros_like(p) for p in points]
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            import pdb; pdb.set_trace()
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            if img_depth is not None:
                loss_depth = self.depth_dist_loss(depth_dist, img_depth, loss_method=self.img_depth_loss_method, img=img) * self.img_depth_loss_weight
                losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas, gt_bboxes_3d)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        #points = [torch.zeros_like(p) for p in points]
        #print("careful, lidar is zero!!")

        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list



@DETECTORS.register_module()
class BEVF_TransFusion_SB(BEVF_TransFusion):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(BEVF_TransFusion_SB, self).__init__(**kwargs)
    
    def extract_feat(self, points, img, img_metas, modality, gt_bboxes_3d=None):
        """Extract features from images and points."""
        if modality == "sbnet_lidar":
            pts_feats = self.extract_pts_feat(points, None, None) # img and img_metas are not used in that func
            return self.seblock(pts_feats[0])

        elif modality == "sbnet_camera":
            img_feats = self.extract_img_feat(img, img_metas)
            BN, C, H, W = img_feats[0].shape
            batch_size = BN//self.num_views
            img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W)
            rots = []
            trans = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                for mat in img_metas[sample_idx]['lidar2img']:
                    mat = torch.Tensor(mat).to(img_feats_view.device)
                    rot_list.append(mat.to("cpu").inverse()[:3, :3].to("cuda"))
                    trans_list.append(mat.to("cpu").inverse()[:3, 3].view(-1).to("cuda"))
                rot_list = torch.stack(rot_list, dim=0)
                trans_list = torch.stack(trans_list, dim=0)
                rots.append(rot_list)
                trans.append(trans_list)
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            lidar2img_rt = img_metas[sample_idx]['lidar2img']  #### extrinsic parameters for multi-view images
            img_bev_feat, _ = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas)
            return img_feats, self.seblock(img_bev_feat)
        else:
            raise ValueError("No SBNet modality found in img_metas")

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        
        modality = img_metas[0]['sbnet_modality']
        if modality == "both":
            feature_dict = self.extract_feat(points, img=img, img_metas=img_metas, modality="camera")
            feature_dict = self.extract_feat(points, img=img, img_metas=img_metas, modality="lidar")
            raise ValueError("fusion mechansim in training to be implemented")
        elif modality == "lidar": 
            pts_feats = self.extract_feat(points, img=img, img_metas=img_metas, modality=modality)
        elif modality == "camera":
            img_feat_2d, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas, modality=modality) # calling it pts_feats bc to avoid specifying forward_pts_train twice
        else:
            raise ValueError("No SBNet modality found in img_metas")
        
        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, None, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore) #  pos 2: img_feats is replaced with None. bc when transfusion_head fuse_img = False then img_feats is not needed. but it does need img metas for box_type_3d (afaik only that but need to learn more)
        losses.update(losses_pts)
        
        if modality == "camera":
            losses_img = self.forward_img_train(
                img_feat_2d,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            # img_depth is always None afaik
            #if img_depth is not None:
            #    loss_depth = self.depth_dist_loss(depth_dist, img_depth, loss_method=self.img_depth_loss_method, img=img) * self.img_depth_loss_weight
            #    losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses
