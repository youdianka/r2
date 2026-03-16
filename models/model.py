import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18
from models.pvt_v2 import PyramidVisionTransformerV2, pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from models.vmamba import VSSM
import os
import torch
from einops import rearrange, reduce, repeat
from models.vmamba import SS2D, VSSBlock


class UP(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UP, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, dilation=dilation),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


class PixelFeatureExchange(nn.Module):

    def __init__(self, group_size=3):
        super(PixelFeatureExchange, self).__init__()
        self.group_size = group_size

    def forward(self, x_list):

        # 确保输入是列表且包含相同数量的特征图
        if not isinstance(x_list, list):
            raise TypeError("输入必须是张量列表")

        num_features = len(x_list)
        if num_features != self.group_size:
            raise ValueError(f"输入特征图数量必须为{self.group_size}，但得到{num_features}")

        # 获取特征图尺寸
        batch_size, channels, height, width = x_list[0].shape

        # 确保所有特征图尺寸一致
        for i, feat in enumerate(x_list):
            if feat.shape != x_list[0].shape:
                raise ValueError(f"特征图 {i} 的尺寸 {feat.shape} 与第一个特征图 {x_list[0].shape} 不一致")

        # 1. 行交换
        row_exchanged = self._exchange_rows(x_list, height, width)

        # 2. 列交换
        final_features = self._exchange_columns(row_exchanged, height, width)

        return final_features

    def _exchange_rows(self, features, height, width):

        result = [torch.zeros_like(feat) for feat in features]
        num_features = len(features)

        # 处理每个行组
        for row_start in range(0, height, self.group_size):
            row_end = min(row_start + self.group_size, height)

            # 对当前行组内的每一行进行交换
            for i in range(row_start, row_end):
                row_idx = i - row_start  # 当前行在组内的索引

                # 确定源特征图索引
                src_feat_idx = row_idx % num_features

                # 为每个目标特征图分配对应的行
                for dst_feat_idx in range(num_features):
                    # 计算从哪个特征图获取行
                    src_idx = (dst_feat_idx + src_feat_idx) % num_features

                    # 复制行数据
                    result[dst_feat_idx][:, :, i, :] = features[src_idx][:, :, i, :]

        return result

    def _exchange_columns(self, features, height, width):

        result = [torch.zeros_like(feat) for feat in features]
        num_features = len(features)

        # 处理每个列组
        for col_start in range(0, width, self.group_size):
            col_end = min(col_start + self.group_size, width)

            # 对当前列组内的每一列进行交换
            for j in range(col_start, col_end):
                col_idx = j - col_start  # 当前列在组内的索引

                # 确定源特征图索引
                src_feat_idx = col_idx % num_features

                # 为每个目标特征图分配对应的列
                for dst_feat_idx in range(num_features):
                    # 计算从哪个特征图获取列
                    src_idx = (dst_feat_idx + src_feat_idx) % num_features

                    # 复制列数据
                    result[dst_feat_idx][:, :, :, j] = features[src_idx][:, :, :, j]

        return result


class SelfAttention(nn.Module):

    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()

        # [B, C', H*W]
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        # [B, C', H*W]
        proj_key = self.key(x).view(batch_size, -1, height * width)
        # [B, H*W, H*W]
        energy = torch.bmm(proj_query, proj_key)
        # [B, H*W, H*W]
        attention = self.softmax(energy)
        # [B, C, H*W]
        proj_value = self.value(x).view(batch_size, -1, height * width)
        # [B, C, H*W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # [B, C, H, W]
        out = out.view(batch_size, C, height, width)

        out = self.gamma * out + x
        return out


class PWConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(PWConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialEnhancement(nn.Module):

    def __init__(self, channels):
        super(SpatialEnhancement, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        mask = self.sigmoid(out)
        return x * mask


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,  # 3×3卷积核
            stride=1,
            padding=1,  # 保持特征图尺寸
            bias=False  # 如果后面接BN层，通常设为False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWConv3x3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DWConv3x3, self).__init__()
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DualBranchAttentiveFusion(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels):
        super(DualBranchAttentiveFusion, self).__init__()

        # 初始PWC卷积
        self.pwc_in = PWConv(in_channels1 + in_channels2, out_channels * 2)

        # 空间增强模块
        self.se = SpatialEnhancement(out_channels)

        # 通道注意力模块
        self.ca = ChannelAttention(out_channels)

        # 最终PWC卷积
        self.pwc_out = PWConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        # 确保特征尺寸一致
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # 拼接输入特征
        concat_in = torch.cat([x1, x2], dim=1)

        # 通过第一个PWC卷积
        feat = self.pwc_in(concat_in)

        # 分割特征
        feat1, feat2 = torch.chunk(feat, 2, dim=1)

        # 并行处理：SE和CA
        feat1_se = self.se(feat1) + feat1  # SE处理 + 残差连接
        feat2_ca = self.ca(feat2) + feat2  # CA处理 + 残差连接

        # 拼接处理后的特征
        concat_out = torch.cat([feat1_se, feat2_ca], dim=1)

        # 最终PWC卷积
        out = self.pwc_out(concat_out)

        return out


class HierarchicalGatedMultiScaleAggregation(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super(HierarchicalGatedMultiScaleAggregation, self).__init__()

        # 对每个输入的处理，确保通道数一致
        self.pwc1 = PWConv(in_channels_list[0], out_channels)  # 处理浅层特征 (64->out_channels)
        self.pwc2 = PWConv(in_channels_list[1], out_channels)  # 处理中层特征 (128->out_channels)
        self.pwc3 = PWConv(in_channels_list[2], out_channels)  # 处理深层特征 (256->out_channels)

        # 上采样模块
        self.up1 = UP(out_channels, out_channels)  # f3 -> f2 大小
        self.up2 = UP(out_channels, out_channels)  # f3f2_conv -> f1 大小

        # 第一次融合后的3×3卷积
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dwconv3x3 = DWConv3x3(out_channels, out_channels)

    def forward(self, x1, x2, x3):
        # 特征转换，确保通道数一致
        f1 = self.pwc1(x1)  # 浅层特征 (64->out_channels)
        f2 = self.pwc2(x2)  # 中层特征 (128->out_channels)
        f3 = self.pwc3(x3)  # 深层特征 (256->out_channels)

        # 后续融合逻辑保持不变
        # 第一步：将f3上采样到f2的大小
        f3_up = self.up1(f3)

        # 如果上采样后的尺寸与f2不一致，进行调整
        if f3_up.shape[2:] != f2.shape[2:]:
            f3_up = F.interpolate(f3_up, size=f2.shape[2:], mode='bilinear', align_corners=True)

        # 第二步：f3_up和f2相乘
        f3f2_mul = f3_up * f2
        f3f2_mul = self.conv3x3(f3f2_mul)
        # 第三步：乘积结果与f3_up相加
        f3f2_add = f3f2_mul + f3_up

        # 第四步：通过3×3卷积
        f3f2_conv = self.conv3x3(f3f2_add)

        # 第五步：上采样到f1的大小
        f3f2_up = self.up2(f3f2_conv)

        # 如果上采样后的尺寸与f1不一致，进行调整
        if f3f2_up.shape[2:] != f1.shape[2:]:
            f3f2_up = F.interpolate(f3f2_up, size=f1.shape[2:], mode='bilinear', align_corners=True)

        # 第六步：上采样结果与f1相乘
        f3f2f1_mul = f3f2_up * f1
        f3f2f1_mul = self.conv3x3(f3f2f1_mul)
        # 第七步：乘积结果与上采样结果相加
        f3f2f1_add = f3f2f1_mul + f3f2_up + f1

        # 第八步：通过3×3卷积得到最终输出
        out = self.dwconv3x3(f3f2f1_add)

        return out


class PixelExchangeGuidedStateSpaceInteractionFusion(nn.Module):

    def __init__(self, input_channels, output_channels, d_state=16, dropout=0.1):
        super(PixelExchangeGuidedStateSpaceInteractionFusion, self).__init__()

        assert len(input_channels) == 3, "Expected 3 channel dimensions in input_channels"
        self.input_channels = input_channels

        # 内部统一通道数
        self.hidden_dim = 64

        # 通道对齐：把不同 stage 的通道统一成 hidden_dim
        self.channel_align = nn.ModuleList([
            nn.Conv2d(input_channels[i], self.hidden_dim, kernel_size=1)
            for i in range(3)
        ])

        # 像素级特征交换
        self.pix_exchange = PixelFeatureExchange(group_size=3)

        # 三个 VSS 分支
        self.vss1 = VSSBlock(hidden_dim=self.hidden_dim, d_state=d_state, dropout=dropout)
        self.vss2 = VSSBlock(hidden_dim=self.hidden_dim, d_state=d_state, dropout=dropout)
        self.vss3 = VSSBlock(hidden_dim=self.hidden_dim, d_state=d_state, dropout=dropout)

        # K/Q/V 映射
        # 分支1、2：只产生 K
        self.to_k1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.to_k2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        # 分支3：产生 Q 和 V
        self.to_q3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.to_v3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)

        # 输出投影：用 3×3 做融合（你前面说想用 3×3）
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 3, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, red_outputs):
        # red_outputs: [red_feat1, red_feat2, red_feat3]
        assert len(red_outputs) == 3, "Expected 3 inputs from red modules"

        # 1. 空间尺寸对齐 + 通道对齐到 hidden_dim
        target_size = red_outputs[0].shape[2:]
        aligned_features = []
        for i, feat in enumerate(red_outputs):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feat = self.channel_align[i](feat)  # [B, hidden_dim, H, W]
            aligned_features.append(feat)

        # 2. PixelFeatureExchange
        exchanged_features = self.pix_exchange(aligned_features)

        # 3. 三个分支分别通过 VSSBlock
        vss_outputs = []
        for i, feat in enumerate(exchanged_features):
            feat_hwc = feat.permute(0, 2, 3, 1)  # BCHW -> BHWC
            if i == 0:
                vss_out = self.vss1(feat_hwc)
            elif i == 1:
                vss_out = self.vss2(feat_hwc)
            else:
                vss_out = self.vss3(feat_hwc)
            vss_out = vss_out.permute(0, 3, 1, 2)  # BHWC -> BCHW
            vss_outputs.append(vss_out)

        # 三个 VSS 分支输出: 每个 [B, C, H, W], C=hidden_dim
        vss1_out, vss2_out, vss3_out = vss_outputs

        B, C, H, W = vss1_out.shape
        HW = H * W

        # 4. 生成 K1, K2, Q3, V3
        # 分支1/2：K
        K1 = self.to_k1(vss1_out)   # [B, C, H, W]
        K2 = self.to_k2(vss2_out)   # [B, C, H, W]
        # 分支3：Q 和 V
        Q3 = self.to_q3(vss3_out)   # [B, C, H, W]
        V3 = self.to_v3(vss3_out)   # [B, C, H, W]

        # reshape 成注意力计算的常用形式
        # Q3: [B, C, H, W] -> [B, HW, C]
        Q_flat = Q3.view(B, C, HW).permute(0, 2, 1)    # [B, HW, C]

        # K1/K2: [B, C, H, W] -> [B, C, HW]
        K1_flat = K1.view(B, C, HW)                    # [B, C, HW]
        K2_flat = K2.view(B, C, HW)                    # [B, C, HW]

        # V3: [B, C, H, W] -> [B, HW, C]
        V3_flat = V3.view(B, C, HW).permute(0, 2, 1)   # [B, HW, C]

        # 5. 用同一个 (Q3, V3)，分别和 K1、K2 做注意力
        # 分支1注意力: O1 = softmax(Q3 K1) V3
        attn_logits1 = torch.bmm(Q_flat, K1_flat) / (C ** 0.5)  # [B, HW, HW]
        attn1 = F.softmax(attn_logits1, dim=-1)                 # [B, HW, HW]
        O1_flat = torch.bmm(attn1, V3_flat)                     # [B, HW, C]
        O1 = O1_flat.permute(0, 2, 1).view(B, C, H, W)          # [B, C, H, W]

        # 分支2注意力: O2 = softmax(Q3 K2) V3
        attn_logits2 = torch.bmm(Q_flat, K2_flat) / (C ** 0.5)  # [B, HW, HW]
        attn2 = F.softmax(attn_logits2, dim=-1)                 # [B, HW, HW]
        O2_flat = torch.bmm(attn2, V3_flat)                     # [B, HW, C]
        O2 = O2_flat.permute(0, 2, 1).view(B, C, H, W)          # [B, C, H, W]

        # 6. 求和：两个注意力输出 + 第三分支原始特征
        #    F_out = O1 + O2 + vss3_out
        fused = O1 + O2     # [B, C, H, W]
        concat_feat = torch.cat([fused, vss1_out, vss2_out], dim=1)

        # 7. 输出投影到 output_channels
        output = self.output_proj(concat_feat)  # [B, output_channels, H, W]

        return output




class BaseNet(nn.Module):
    def __init__(self,
                 channel=96,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9],
                 depths_decoder=[2, 9, 2],
                 drop_path_rate=0.2,
                 pvt_version='b2', resnet_pretrained=True,  # 添加这个参数
                 pvt_pretrained_path='/userA02/lyp/pretrained_pth/pvt_v2_b2.pth',
                 mamba_pretrained_path='/userA02/lyp/pretrained_pth/vmamba_small_e238_ema.pth'
                 ):
        super(BaseNet, self).__init__()
        # 第一个骨干：ResNet18
        self.cnn_backbone = resnet18(pretrained=True)

        # 第二个骨干：VSSM
        self.vssm_backbone = VSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate
        )

        # 第三个骨干：PVTv2
        if pvt_version == 'b0':
            self.pvt_backbone = pvt_v2_b0(pretrained=True)
        elif pvt_version == 'b1':
            self.pvt_backbone = pvt_v2_b1(pretrained=True)
        elif pvt_version == 'b2':
            self.pvt_backbone = pvt_v2_b2(pretrained=True)
        elif pvt_version == 'b3':
            self.pvt_backbone = pvt_v2_b3(pretrained=True)
        elif pvt_version == 'b4':
            self.pvt_backbone = pvt_v2_b4(pretrained=True)
        elif pvt_version == 'b5':
            self.pvt_backbone = pvt_v2_b5(pretrained=True)
        else:
            raise ValueError(f"Unsupported PVTv2 version: {pvt_version}")
        self._load_pretrained_weights(resnet_pretrained, pvt_pretrained_path, mamba_pretrained_path)
        self.res_ch = [64, 128, 256]  # ResNet各阶段通道数
        self.pvt_ch = [64, 128, 320]  # PVT各阶段通道数
        self.vssm_ch = [96, 192, 384]  # VSSM (Mamba)各阶段通道数
        self.purple_fusion_out_ch = [64, 128, 256]  # 各紫色融合模块的输出通道数
        self.red_fusion_out_ch = [64, 128, 256]
        channel = 64
        # 定义紫色融合模块 (PurpleFusion)
        # 第一层紫色模块 (CNN + Transformer)
        # Purple (DAF)
        self.purple_fusion1_1 = DualBranchAttentiveFusion(self.res_ch[0], self.pvt_ch[0], self.purple_fusion_out_ch[0])
        self.purple_fusion2_1 = DualBranchAttentiveFusion(self.res_ch[1], self.pvt_ch[1], self.purple_fusion_out_ch[1])
        self.purple_fusion3_1 = DualBranchAttentiveFusion(self.res_ch[2], self.pvt_ch[2], self.purple_fusion_out_ch[2])

        self.purple_fusion1_2 = DualBranchAttentiveFusion(self.res_ch[0], self.vssm_ch[0], self.purple_fusion_out_ch[0])
        self.purple_fusion2_2 = DualBranchAttentiveFusion(self.res_ch[1], self.vssm_ch[1], self.purple_fusion_out_ch[1])
        self.purple_fusion3_2 = DualBranchAttentiveFusion(self.res_ch[2], self.vssm_ch[2], self.purple_fusion_out_ch[2])

        self.purple_fusion1_3 = DualBranchAttentiveFusion(self.pvt_ch[0], self.vssm_ch[0], self.purple_fusion_out_ch[0])
        self.purple_fusion2_3 = DualBranchAttentiveFusion(self.pvt_ch[1], self.vssm_ch[1], self.purple_fusion_out_ch[1])
        self.purple_fusion3_3 = DualBranchAttentiveFusion(self.pvt_ch[2], self.vssm_ch[2], self.purple_fusion_out_ch[2])

        # Red (HGMA)
        self.red_fusion1 = HierarchicalGatedMultiScaleAggregation(self.purple_fusion_out_ch, self.red_fusion_out_ch[0])
        self.red_fusion2 = HierarchicalGatedMultiScaleAggregation(self.purple_fusion_out_ch, self.red_fusion_out_ch[1])
        self.red_fusion3 = HierarchicalGatedMultiScaleAggregation(self.purple_fusion_out_ch, self.red_fusion_out_ch[2])

        # Green (PGSIF)
        self.green_module = PixelExchangeGuidedStateSpaceInteractionFusion(self.red_fusion_out_ch, channel, d_state=16,
                                                                           dropout=0.1)

        # 最终预测头
        self.pred_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    def _load_pretrained_weights(self, resnet_pretrained, pvt_path, mamba_path):
        # ResNet权重加载
        if resnet_pretrained:
            import torchvision.models as models
            pretrained_resnet = models.resnet18(pretrained=True)

            # 提取前三个stage的权重
            pretrained_dict = pretrained_resnet.state_dict()
            model_dict = self.cnn_backbone.state_dict()

            # 过滤出需要的权重（排除layer4）
            filtered_dict = {k: v for k, v in pretrained_dict.items()
                             if k in model_dict and not k.startswith('layer4')}

            # 更新模型权重
            model_dict.update(filtered_dict)
            self.cnn_backbone.load_state_dict(model_dict, strict=False)
            print(f"ResNet18 loaded: {len(filtered_dict)}/{len(model_dict)} keys")

        # PVT权重加载
        if pvt_path and os.path.exists(pvt_path):
            try:
                checkpoint = torch.load(pvt_path, map_location='cpu')

                if 'model' in checkpoint:
                    pretrained_dict = checkpoint['model']
                else:
                    pretrained_dict = checkpoint

                model_dict = self.pvt_backbone.state_dict()

                # 打印关键信息以便调试
                print(f"PVT预训练模型参数数量: {len(pretrained_dict)}")
                print(f"当前PVT模型参数数量: {len(model_dict)}")

                # 检查形状不匹配的参数
                mismatch_params = []
                for k in pretrained_dict:
                    if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape:
                        mismatch_params.append((k, pretrained_dict[k].shape, model_dict[k].shape))

                if mismatch_params:
                    print("形状不匹配的参数:")
                    for param, pre_shape, model_shape in mismatch_params:
                        print(f"  - {param}: 预训练形状 {pre_shape}, 当前模型形状 {model_shape}")

                # 排除head和block4相关参数，并确保形状匹配
                filtered_dict = {k: v for k, v in pretrained_dict.items()
                                 if k in model_dict and model_dict[k].shape == v.shape
                                 and not k.startswith('block4') and not k.startswith('head')}

                # 更新模型权重
                model_dict.update(filtered_dict)
                self.pvt_backbone.load_state_dict(model_dict, strict=False)
                print(f"PVT loaded: {len(filtered_dict)}/{len(model_dict)} keys")

            except Exception as e:
                print(f"Error loading PVT weights: {e}")
                import traceback
                traceback.print_exc()  # 打印完整错误堆栈

        # VSSM权重加载
        if mamba_path and os.path.exists(mamba_path):
            try:
                checkpoint = torch.load(mamba_path, map_location='cpu')

                if 'model' in checkpoint:
                    pretrained_dict = checkpoint['model']
                else:
                    pretrained_dict = checkpoint

                model_dict = self.vssm_backbone.state_dict()  # 修正变量名

                # 过滤权重，只保留前三个stage
                filtered_dict = {k: v for k, v in pretrained_dict.items()
                                 if k in model_dict and not any(x in k for x in ['stage4', 'head'])}

                # 更新模型权重
                model_dict.update(filtered_dict)
                self.vssm_backbone.load_state_dict(model_dict, strict=False)  # 修正变量名
                print(f"VSSM loaded: {len(filtered_dict)}/{len(model_dict)} keys")
            except Exception as e:
                print(f"Error loading VSSM weights: {e}")

    def forward(self, x):
        input_size = x.shape[2:]
        # ResNet18 特征提取
        cnn_x = self.cnn_backbone.conv1(x)
        cnn_x = self.cnn_backbone.bn1(cnn_x)
        cnn_x = self.cnn_backbone.relu(cnn_x)
        cnn_x = self.cnn_backbone.maxpool(cnn_x)

        # 第一阶段特征
        cnn_feat1 = self.cnn_backbone.layer1(cnn_x)  # 1/4, 64
        # 第二阶段特征
        cnn_feat2 = self.cnn_backbone.layer2(cnn_feat1)  # 1/8, 128
        # 第三阶段特征
        cnn_feat3 = self.cnn_backbone.layer3(cnn_feat2)  # 1/16, 256

        # PVTv2 特征提取
        pvt_feats = self.pvt_backbone(x)  # 返回多尺度特征列表
        pvt_feat1 = pvt_feats[0]  # 1/4, 64
        pvt_feat2 = pvt_feats[1]  # 1/8, 128
        pvt_feat3 = pvt_feats[2]  # 1/16, 320

        # VSSM (Mamba) 特征提取
        vssm_feat1, vssm_feat2, vssm_feat3 = self.vssm_backbone(x)  # 返回三个阶段的特征
        # 注意：VSSM的特征是BHWC格式，需要转换为BCHW格式
        vssm_feat1 = vssm_feat1.permute(0, 3, 1, 2)  # 1/4, 96
        vssm_feat2 = vssm_feat2.permute(0, 3, 1, 2)  # 1/8, 192
        vssm_feat3 = vssm_feat3.permute(0, 3, 1, 2)  # 1/16, 384

        # CNN + Transformer 融合
        purple_feat1_1 = self.purple_fusion1_1(cnn_feat1, pvt_feat1)  # 1/4
        purple_feat2_1 = self.purple_fusion2_1(cnn_feat2, pvt_feat2)  # 1/8
        purple_feat3_1 = self.purple_fusion3_1(cnn_feat3, pvt_feat3)  # 1/16

        # CNN + Mamba 融合
        purple_feat1_2 = self.purple_fusion1_2(cnn_feat1, vssm_feat1)  # 1/4
        purple_feat2_2 = self.purple_fusion2_2(cnn_feat2, vssm_feat2)  # 1/8
        purple_feat3_2 = self.purple_fusion3_2(cnn_feat3, vssm_feat3)  # 1/16

        # Transformer + Mamba 融合
        purple_feat1_3 = self.purple_fusion1_3(pvt_feat1, vssm_feat1)  # 1/4
        purple_feat2_3 = self.purple_fusion2_3(pvt_feat2, vssm_feat2)  # 1/8
        purple_feat3_3 = self.purple_fusion3_3(pvt_feat3, vssm_feat3)  # 1/16

        red_feat1 = self.red_fusion1(purple_feat1_1, purple_feat2_1, purple_feat3_1)  # 1/4
        red_feat2 = self.red_fusion2(purple_feat1_2, purple_feat2_2, purple_feat3_2)  # 1/8
        red_feat3 = self.red_fusion3(purple_feat1_3, purple_feat2_3, purple_feat3_3)  # 1/16

        green_feat = self.green_module([red_feat1, red_feat2, red_feat3])  # 1/4

        pred = self.pred_head(green_feat)
        if pred.shape[2:] != input_size:
            pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
        return pred
