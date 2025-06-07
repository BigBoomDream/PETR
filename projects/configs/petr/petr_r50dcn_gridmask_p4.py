_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'  # 这里引用了nus-3d的nuscenes数据集，所以包含了在mm3d中的配置，default_runtime是基本的runtime设置。
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# 使用均值和标准差进行图像归一化
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], 
    std=[1.0, 1.0, 1.0], 
    to_rgb=False)   # 表示输入图像为BGR格式

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
model = dict(
    type='Petr3D',
    use_grid_mask=True,
    '''
        首先是backbone，是一个resnet50，输入数据维度（B，N，3，H，W），查看源码后发现如果是5维的tensor，会将BN相乘后转换到4维输入。
    '''
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3,),    # （0索引）输出第3，4层的中间特征，维度为1024，2048，对应FPN网络
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,     # 在训练期间将BN层设置为评估模式（使用运行均值/方差）
        style='caffe',
        with_cp=True,   # 为主干网络启用检查点机制。这是一种节省内存的技术，在正向传播期间不存储中间激活，而是在反向传播期间重新计算。它用计算换取内存。
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),   # 后两个阶段使用DCN（可变性卷积）
        stage_with_dcn=(False, False, True, True),
        pretrained = 'ckpts/resnet50_msra-5891d200.pth',
        ),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],   # ResNet-50的第3阶段（索引2）输出有1024个通道，第4阶段（索引3）输出有2048个通道。
        out_channels=256,
        num_outs=2),    #  FPN输出的特征图数量。这对应于 img_backbone.out_indices 的两个输入特征级别。   
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=10,
        in_channels=256,    # 输入到“Detr3DHead”的通道数，就是img_neck的输出通道数。
        num_query=900,
        LID=True,   # 线性递增离散化 (LID)
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,   # 每一层都要得到预测结果，当作辅助损失
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,  # FFN前馈神经网络的中间层维度
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            # type='NMSFreeClsCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding3D',    # 自定义的位置编码 projects/mmdet3d_plugin/models/utils/positional_encoding.py
            num_feats=128,      # x、y和z坐标中每个坐标的特征维度数。
            normalize=True),    # 如果为true，则在应用正弦/余弦函数之前对输入坐标进行归一化。
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),

    # train_cfg定义了在Detr3DHead训练时的配置
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        # assigner 将预测的结果与GT进行一一匹配
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'  # projects/mmdet3d_plugin/datasets/nuscenes_dataset.py
data_root = '/data/Dataset/nuScenes/'

file_client_args = dict(backend='disk')
db_sampler = dict() # 在DETR3D中， Database Sampler 数据增强方法，存储了训练集中的GT，然后在训练一个新场景时，随机挑出一些物体，把他们粘贴到当前正在处理的场景中

# 图像数据增强 (IDA) 的配置，特定于BEVDet风格的增强
ida_aug_conf = {
        "resize_lim": (0.8, 1.0),   # 输入图像随机调整大小的范围。
        "final_dim": (512, 1408),   # 图像调整大小/裁剪后的最终尺寸（高，宽）
        "bot_pct_lim": (0.0, 0.0),  # 定义图像底部可能被裁剪的百分比。(0.0, 0.0) 表示不进行底部裁剪。
        "rot_lim": (0.0, 0.0),      # 随机旋转的角度范围（度）。(0.0, 0.0) 表示不旋转。
        "H": 900,   # nuScenes相机的原始图像尺寸
        "W": 1600,  
        "rand_flip": True,          # 启用图像的随机水平翻转
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),    # 过滤掉中心点在定义的 point_cloud_range 之外的真实对象。
    dict(type='ObjectNameFilter', classes=class_names),     # 过滤掉类别名称不在提供的 class_names 中的真实对象。
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),      # Todo  DETR3D中没有的！  应用 ida_aug_conf 中定义的调整大小、裁剪和翻转增强。
    dict(type='GlobalRotScaleTransImage',    #Todo  DETR3D中没有的！  对整个3D场景应用全局旋转、缩放和平移增强，这会影响图像和相机参数。 自定义projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py
            rot_range=[-0.3925, 0.3925],    # 随机旋转范围（弧度）（约-22.5到22.5度）
            translation_std=[0, 0, 0],      # 随机平移的标准差（此处禁用）
            scale_ratio_range=[0.95, 1.05], # 场景随机缩放的范围
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',     # 测试时的数据增强
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,      # 每个GPU每次迭代处理的样本数
    workers_per_gpu=4,      # 每个GPU的数据加载工作线程数    
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,    # 用于训练，对于val/test则隐式为 True。
        use_valid_flag=True,    # 在训练期间使用一个标志来过滤掉无效的注释
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names, modality=input_modality))

optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

'''
    启用混合精度 (FP16) 训练;
    loss_scale=512.: FP16训练的初始损失缩放因子，以防止下溢。
    grad_clip=dict(max_norm=35, norm_type=2): 梯度裁剪以防止梯度爆炸。如果梯度的L2范数 (norm_type=2) 超过 max_norm=35，则进行裁剪。
'''
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))    

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    # by_epoch=False
    )

total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)
find_unused_parameters = False  # 这与分布式数据并行 (DDP) 相关。如果设置为 True，DDP将尝试查找在前向传播中未使用的参数，这有助于调试但会增加开销。在使用检查点机制 (with_cp=True) 时，find_unused_parameters 必须为 False。


runner = dict(type='EpochBasedRunner', max_epochs=total_epochs) # 指定基于周期的训练循环。
load_from=None  # 在训练开始时从中加载权重的检查点文件路径
resume_from=None    # 从特定状态（优化器状态、周期等）恢复训练的检查点文件路径

# mAP: 0.3174
# mATE: 0.8397
# mASE: 0.2796
# mAOE: 0.6158
# mAVE: 0.9543
# mAAE: 0.2326
# NDS: 0.3665
# Eval time: 199.1s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.503   0.607   0.155   0.120   1.107   0.241
# truck   0.259   0.874   0.232   0.217   0.968   0.261
# bus     0.329   0.864   0.219   0.188   2.289   0.411
# trailer 0.105   1.143   0.253   0.548   0.400   0.104
# construction_vehicle    0.071   1.233   0.503   1.216   0.122   0.349
# pedestrian      0.407   0.735   0.294   1.016   0.770   0.313
# motorcycle      0.294   0.810   0.277   0.901   1.471   0.146
# bicycle 0.290   0.698   0.260   1.176   0.509   0.036
# traffic_cone    0.497   0.608   0.322   nan     nan     nan
# barrier 0.419   0.824   0.281   0.160   nan     nan
