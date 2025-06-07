# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
import numpy as np
from mmcv.cnn import xavier_init, constant_init, kaiming_init
import math
from mmdet.models.utils import NormedLinear


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    # self.reference_points = nn.Embedding(self.num_query, 3)  num_query=900
    # pos = self.reference_points.weight  (900,3)
    scale = 2 * math.pi # 缩放因子，通常是 2π。 因为正弦和余弦函数的周期是 2π。将输入坐标乘以这个因子，可以使后续在应用三角函数时，坐标的变换能在一个或多个周期内体现
    pos = pos * scale # 输入的原始坐标pos(900,3) 的每个分量（x, y, z）乘以 scale；是为了将坐标映射到一个更适合三角函数编码的范围。如果原始坐标是归一化到[0,1]的，那乘以 2π 后就映射到了 [0, 2*pi] 区间
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device) # [0., 1., ..., 127.]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t   # [num_queries, 128]
    pos_y = pos[..., 1, None] / dim_t   
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)   # [num_queries, 128]
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)   # 三个坐标维度（y, x, z） [num_queries, 128*3=384]
    return posemb

@HEADS.register_module()
class PETRHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is PETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
            #     'The regression iou weight for loss and matcher should be' \
            #     'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num  # 64
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range    # position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        self.LID = LID  # 线性递增离散化 (LID)
        self.depth_start = depth_start  # 1
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(PETRHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        # self.activate = build_activation_layer(self.act_cfg)
        # if self.with_multiview or not self.with_position:
        #     self.positional_encoding = build_positional_encoding(
        #         positional_encoding)
        self.positional_encoding = build_positional_encoding(
                positional_encoding)
        self.transformer = build_transformer(transformer)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),    # self.position_dim = 3 * self.depth_num(64)=192   self.embed_dims = 256
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, img_feats, img_metas, masks=None):
        '''
            输入的参数：
                img_feats, 经过backbone+fpn之后的多尺度特征
                img_metas,  
                masks 告诉模型哪些区域是图像的有效内容 (值为0)，哪些是填充区域 (值为1，应该被忽略)。
        '''
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape    # self.position_level = 0    
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H      #  pad_h 填充的目的是能被32整除，因此pad_h可能比原图片尺寸要大， 
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W      #  pad_w / W 就是计算采样了多少层，这里应该是1/16
        '''
            原图(1600*900)
            填充图像 (1632 × 928):
            ┌─────────────────────────────┐
            │ ● ● ● ● ● ● ● ● ● ● ● ● ● ●│  像素级密集采样
            │ ● ● ● ● ● ● ● ● ● ● ● ● ● ●│
            │ ● ● ● ● ● ● ● ● ● ● ● ● ● ●│
            └─────────────────────────────┘

            特征图 (408×232):
            ┌───────────────┐
            │ ▲   ▲   ▲   ▲ │  每4个像素一个特征点
            │               │
            │ ▲   ▲   ▲   ▲ │
            └───────────────┘

            coords_h, coords_w 建立映射关系:
            特征图位置 (0,0) → 原图位置 (0,0)
            特征图位置 (1,1) → 原图位置 (4,4)  
            特征图位置 (2,2) → 原图位置 (8,8)
        '''

        if self.LID:    # 线性递增离散化 (LID)
            #  self.depth_num = depth_num  # depth_num = 64          
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()  # index = [0, 1, 2, 3, ..., 63]      # 64个索引
            index_1 = index + 1              # index_1 = [1, 2, 3, 4, ..., 64]    
            # position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            # self.depth_start = depth_start  # depth_start = 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))    # bin_size = (61.2 - 1) / (64 * (1 + 64)) = ≈ 0.0145
            '''
                        计算深度值（非线性分布）
                coords_d[0] = 1 + 0.0145 * 0 * 1 = 1.000
                coords_d[1] = 1 + 0.0145 * 1 * 2 = 1.029  
                coords_d[2] = 1 + 0.0145 * 2 * 3 = 1.087
                coords_d[3] = 1 + 0.0145 * 3 * 4 = 1.174
                ...
                coords_d[63] = 1 + 0.0145 * 63 * 64 ≈ 59.2

                深度  1m    2m    5m    10m   20m   40m   60m
                密度  |||||  ||||   |||    ||     |     |    |
            '''
            coords_d = self.depth_start + bin_size * index * index_1
        else: # 均匀分布
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num         # bin_size = (61.2 - 1) / 64 = 60.2 / 64 ≈ 0.941
            '''
                         均匀分布
                    coords_d[0] = 1 + 0.941 * 0 = 1.000
                    coords_d[1] = 1 + 0.941 * 1 = 1.941
                    coords_d[2] = 1 + 0.941 * 2 = 2.882
                    coords_d[3] = 1 + 0.941 * 3 = 3.823
                    ...
                    coords_d[63] = 1 + 0.941 * 63 ≈ 60.3
            '''
            coords_d = self.depth_start + bin_size * index  


        D = coords_d.shape[0]   # D = 64

        '''
            coords_w.shape = [w]  # 宽度方向坐标
            coords_h.shape = [h]  # 高度方向坐标  
            coords_d.shape = [d]   # 深度方向坐标
            meshgrid生成3D网格: meshgrid_result = torch.meshgrid([coords_w, coords_h, coords_d]) # 结果是3个张量，每个形状为 [232, 408, 64]
            stack后形状变为 [3, W, H, D]   permute(1,2,3,0)后变为 [W, H, D, 3]
            coords.shape = [W, H, D, 3]  # 每个位置存储(x, y, z)坐标
        '''
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3  总共采样了w*h*d个3D采样点
        
        # 从 [W, H, D, 3] 变为 [W, H, D, 4]，齐次坐标； 每个点从 (x, y, z) 变为 (x, y, z, 1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)      # [W, H, D, 4]
        # 透视投影校正：近处的点：深度z较小，x,y坐标几乎不变；   远处的点：深度z较大，x,y坐标按比例放大
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)  # 刚好和DETR3d里面的消除透视反过来
        
        # --- 构建 图像-> 世界坐标系的逆变换矩阵 ----
        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                # np.linalg.inv 用于计算矩阵的逆的函数
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))   # 通过求逆矩阵得到 img2lidar
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars) # (B, N, 4, 4)   # 将NumPy数组转换为PyTorch张量，并确保与coords张量具有相同的设备位置和数据类型。

        # --- 反投影到世界坐标系下，然后归一化坐标 ----
        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)   # [B, N=6, W, H, D=64, 4, 1]
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)   # [B, N=6, W, H, D=64, 4, 4]
        '''
            对每个图像像素进行坐标转换
            img2lidars: [B, N, W, H, D, 4, 4]
            coords:     [B, N, W, H, D, 4, 1]
            结果:       [B, N, W, H, D, 4, 1]
         squeeze(-1)[..., :3] 移除最后一维并只保留(X,Y,Z)  coords3d.shape = [B, N, W, H, D, 3]
        '''
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]    # [B, N, W, H, D, 3]
        # 归一化：原始坐标: (X=30.6, Y=-30.6, Z=5.0)    归一化后: (X=0.75, Y=0.25, Z=0.75)   [B, N, W, H, D, 3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        # --- 掩码生成 ----
        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)   # [B, N, W, H, D, 3]
        # 如果一个(B,N,W,H)位置，其在所有D=64个深度上的3个坐标(x,y,z)中，有一半以上的坐标值超出了[0,1]的范围，
        # 那么就认为这个(B,N,W,H)位置的3D信息大部分是无效的。
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)   # [B, N, W, H, D, 3] → [B, N, W, H]
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)   # 逻辑或：只要一个像素在2D图像上是填充，或者它对应的3D点大部分超出了范围，那么它对应的位置嵌入就应该被标记为无效。   得到[B, N, H, W]

        # --- 生成位置编码 ----
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)  #  [B, N, W, H, D, 3] → [B, N, D, 3, H, W] → [B*N, D*3, H, W]
        coords3d = inverse_sigmoid(coords3d) # 为了让后续的 position_encoder能够更有效地学习和编码这些位置信息，避免因 sigmoid 函数的饱和区导致的梯度消失问题，从而提高模型学习的稳定性和效率。    [B*N, D*3, H, W]
        '''
            self.position_encoder = nn.Sequential(
                    nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),    # self.position_dim = 3 * self.depth_num(64)=3*D   self.embed_dims = 256
                    nn.ReLU(),
                    nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
                )
        '''
        coords_position_embeding = self.position_encoder(coords3d)  # [B*N, D*3, H, W] ->  [B*N, embed_dims=256, H, W] 
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask  # [B*N, embed_dims=256, H, W] -> (B, N, embed_dims=256, H, W)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is PETRHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
         
        x = mlvl_feats[0]   # mlvl_feats是一个列表，取这个列表的第一个(还是个列表)
        batch_size, num_cams = x.size(0), x.size(1) # x 经过backbone+fpn之后的一层特征 (B, N, C=256, H, W)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]      # 经过数据增强后的，且经过了backbone+fpn产生的特征图的宽、高【有填充的区域】
        
        '''
            假设填充后的图像是 800x600，但原始有效图像区域是 720x540。
            那么 masks 中 [:720, :540] 的区域会是0，其余部分是1。
            有效区域mask值为 0 ，无效区域为 1
        '''
        masks = x.new_ones( 
            (batch_size, num_cams, input_img_h, input_img_w))   # masks 全1的tensor
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]    # 没有填充之前的图像尺寸，也就是真实的图片尺寸，是有效区域的；
                # 一张图片实际是 800×1200，但为了统一输入尺寸被 padding 成了 832×1216，那么 pad 的区域就不是有效图像。
                masks[img_id, cam_id, :img_h, :img_w] = 0

        # self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)        
        x = self.input_proj(x.flatten(0,1)) # flatten(0,1) 将 (B, N, C, H, W) 的特征图展平成 (B*N, C, H, W)，这样可以一次性通过一个2D卷积层。得到(B×N, C, H, W)
        x = x.view(batch_size, num_cams, *x.shape[-3:])     #  (B, N, embed_dims, H, W)

        # interpolate masks to have the same spatial shape with x   
        # 让 mask 与 feature map 对齐
        #    原因：Transformer 的注意力机制是在特征图 x_feat 上操作的。
        #          为了让 Transformer 忽略掉那些由原始图像填充区域产生的无效特征，
        #          我们需要一个与 x_feat 空间维度完全对应的掩码。
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)    

        # --- ✳✳✳✳✳✳✳✳✳✳✳✳ 生成3D位置嵌入 ✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳✳ ---
        if self.with_position:  # 表示显式使用3D位置信息来生成位置感知特征
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)  # coords_position_embeding (B, N, embed_dims=256, H, W)
            pos_embed = coords_position_embeding    # (B, N, embed_dims=256, H, W)
            if self.with_multiview: # with_multiview为true  （多视图统一处理）
                # 直接对所有视图处理 
                # positional_encoding在配置文件中有，projects/mmdet3d_plugin/models/utils/positional_encoding.py
                # sin_embed (batch_size, num_cams, 3 * num_feats, H_feat, W_feat)
                sin_embed = self.positional_encoding(masks) # mask标记特征图的有效图像内容，基于mask生成2D位置编码  [B, N, H, W] -> sin_embed [B, N, embed_dims*3//2, H, W]
                '''
                        self.adapt_pos3d = nn.Sequential(
                            nn.Conv2d(self.embed_dims*3//2 = 384, self.embed_dims*4 = 1024, kernel_size=1, stride=1, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
                        )
                '''
                # sin_embed.flatten(0, 1)   [B*N, embed_dims*3//2 = 384, H, W] -> [B*N, 256, H, W] -> [B, N, 256, H, W]
                # 通过 adapt_pos3d 将2D编码映射到与3D编码相同的维度空间
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())        #  [B, N, 256, H, W]  
                pos_embed = pos_embed + sin_embed   # 3D + 2D 位置编码  [B, N, 256, H, W]  
            else:   
                # with_multiview = False（逐相机处理）
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        # self.reference_points = nn.Embedding(self.num_query, 3)  num_query=900
        reference_points = self.reference_points.weight
        '''
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims*3//2 = 384, self.embed_dims = 256),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        '''
        # pos2posemb3d(reference_points) [num_queries, 128*3=384] -> query_embeds [num_queries,embed_dims=256]
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))     # [num_queries,embed_dims=256]
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)   #.sigmoid()   [batch_size, num_query, 3]

        '''
            x: 特征图(B, N, C=256, H, W)
            masks: 对应特征图的有效特征(B, N, H, W)
            query_embeds: object query，用于生成解码器的初始查询的位置部分 [num_queries,embed_dims=256]
            pos_embed: 对应特征图的位置编码 [B, N, 256, H, W]  
            self.reg_branches:
        '''
        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches) # [num_layers, bs, num_query, dim]
        outs_dec = torch.nan_to_num(outs_dec)   # [num_layers, bs, num_query, dim]
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            # reference_points作为当前层的初始参考点
            reference = inverse_sigmoid(reference_points.clone())  # (B, num_query, 3)
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])  # (B, num_query, num_classes+1)
            tmp = self.reg_branches[lvl](outs_dec[lvl])     # (B, num_query, code_size)

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)   
            outputs_coords.append(outputs_coord)   

        all_cls_scores = torch.stack(outputs_classes)   # (num_layers, B, num_query, num_classes+1)
        all_bbox_preds = torch.stack(outputs_coords)    # (num_layers, B, num_query, code_size)

        # === 将归一化的边界框预测反归一化到实际物理尺度 ===
        # pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        # all_bbox_preds[..., 0:1] 是中心点 x 坐标 (cx)
        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,   # (num_layers, B, num_query, num_classes+1)
            'all_bbox_preds': all_bbox_preds,   # (num_layers, B, num_query, code_size)
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list