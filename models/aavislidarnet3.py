import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    M_PosAttention2d,
    M_PosAttention3d,
    S_ChnAttention2d,
    S_ChnAttention3d,
)

__all__ = ["aavislidarnet3"]


def basicBlock2d(in_channel, out_channel, noLastAct=False):
    """Basic block for visual 2d feature extraction.
    Args:
        in_channel: input channel number, int
        out_channel: output channel number, int
        noLastAct: remove the last activate function
    Returns:
        block: a list of modules, (module1, module2, ...)
    """
    block = [
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    ]
    if noLastAct:
        block.pop()
    return block


def basicBlock3d(in_channel, out_channel, layer_num, noLastAct=False):
    """Basic block for LiDAR 3d feature extraction.
    Args:
        in_channel: input channel number, int
        out_channel: output channel number, int
        layer_num: the number of basic block layers
        noLastAct: remove the last activate function
    Returns:
        block: a list of modules, (module1, module2, ...)
    """
    block = []
    for layer in range(layer_num):
        in_channel1 = in_channel if layer == 0 else out_channel
        block.extend(
            [
                nn.Conv3d(in_channel1, out_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]
        )
    if noLastAct:
        block.pop()
    return block


def fusionBlock(method):
    """Fuse visual and LiDAR features.
    Args:
        method: fusion method, can be one of "cat", "linear" or "mlp", which
                respectively means directly concatenation, linear layers or
                multi-layer perception.
    Returns:
        block: the fusion block of nn.Sequential
    """
    method_dict = {
        "cat": nn.Identity(),
        "linear": nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256)),
        "mlp": nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        ),
    }
    if method not in method_dict:
        raise ValueError('`method` must be one of "cat", "linear" and "mlp".')
    return method_dict[method]


# AAVisLidarNet3, fuse 5 in the PPT.
class AAVisLidarNet3(nn.Module):
    """Attention Augmented Visual-LiDAR net"""

    def __init__(self, fusion_block, Lp):
        super(AAVisLidarNet3, self).__init__()
        self.Lp = Lp  # Lp norm

        # visual feature extraction
        self.v_feature1 = nn.Sequential(
            *basicBlock2d(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *basicBlock2d(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.v_feature2 = nn.Sequential(
            *basicBlock2d(64, 64),
            *basicBlock2d(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.v_feature3 = nn.Sequential(
            *basicBlock2d(128, 128),
            *basicBlock2d(128, 128),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # lidar feature extraction
        self.l_feature1 = nn.Sequential(
            *basicBlock3d(1, 32, layer_num=2), nn.AvgPool3d(kernel_size=2, stride=2),
        )
        self.l_feature2 = nn.Sequential(
            *basicBlock3d(32, 64, layer_num=2),
            *basicBlock3d(64, 64, layer_num=2),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        self.l_feature3 = nn.Sequential(
            *basicBlock3d(64, 128, layer_num=2, noLastAct=False),
            nn.BatchNorm3d(128),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

        # visual attention
        self.v_attention_pos1 = M_PosAttention2d(
            64, kernel_size=3, dk=4, dv=16, Nh=1, stride=2, singleHead=True
        )
        self.v_attention_pos2 = M_PosAttention2d(
            128, kernel_size=3, dk=4, dv=16, Nh=1, stride=2, singleHead=True
        )
        self.v_attention_pos3 = M_PosAttention2d(
            128, kernel_size=3, dk=4, dv=16, Nh=1, stride=1, singleHead=True
        )
        self.v_attention_chn1 = S_ChnAttention2d(in_chn=64, out_chn=16)
        self.v_attention_chn2 = S_ChnAttention2d(in_chn=128, out_chn=16)
        self.v_attention_chn3 = S_ChnAttention2d(in_chn=128, out_chn=16)
        self.v_atten_fusion = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # lidar attention
        self.l_attention_pos1 = M_PosAttention3d(
            32, kernel_size=3, dk=4, dv=16, Nh=1, stride=2, singleHead=True
        )
        self.l_attention_pos2 = M_PosAttention3d(
            64, kernel_size=3, dk=4, dv=16, Nh=1, stride=2, singleHead=True
        )
        self.l_attention_pos3 = M_PosAttention3d(
            128, kernel_size=3, dk=4, dv=16, Nh=1, stride=1, singleHead=True
        )
        self.l_attention_chn1 = S_ChnAttention3d(in_chn=32, out_chn=16)
        self.l_attention_chn2 = S_ChnAttention3d(in_chn=64, out_chn=16)
        self.l_attention_chn3 = S_ChnAttention3d(in_chn=128, out_chn=16)
        self.l_atten_fusion = nn.Sequential(
            nn.Conv3d(96, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # visual & LiDAR attention fusion
        self.vl_atten_fusion = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )


    def forward(self, img, pc):
        """This function train the whole network in an end-to-end way. 
        The inputs are image and LiDAR point cloud."""

        # extract feature & atten seperately --> part 1
        img_fea = self.v_feature1(img)
        pc_fea = self.l_feature1(pc)
        img_fea_detach = img_fea.detach()
        pc_fea_detach = pc_fea.detach()
        img_atten_pos1 = self.v_attention_pos1(img_fea_detach)
        img_atten_chn1 = self.v_attention_chn1(img_fea_detach)
        pc_atten_pos1 = self.l_attention_pos1(pc_fea_detach)
        pc_atten_chn1 = self.l_attention_chn1(pc_fea_detach)

        # extract feature & atten seperately --> part 2
        img_fea = self.v_feature2(img_fea)
        pc_fea = self.l_feature2(pc_fea)
        img_fea_detach = img_fea.detach()
        pc_fea_detach = pc_fea.detach()
        img_atten_pos2 = self.v_attention_pos2(img_fea_detach)
        img_atten_chn2 = self.v_attention_chn2(img_fea_detach)
        pc_atten_pos2 = self.l_attention_pos2(pc_fea_detach)
        pc_atten_chn2 = self.l_attention_chn2(pc_fea_detach)

        # extract feature & atten seperately --> part 3
        img_fea = self.v_feature3(img_fea)
        pc_fea = self.l_feature3(pc_fea)
        img_fea_detach = img_fea.detach()
        pc_fea_detach = pc_fea.detach()
        img_atten_pos3 = self.v_attention_pos3(img_fea_detach)
        img_atten_chn3 = self.v_attention_chn3(img_fea_detach)
        pc_atten_pos3 = self.l_attention_pos3(pc_fea_detach)
        pc_atten_chn3 = self.l_attention_chn3(pc_fea_detach)

        # fuse image attention
        final_sz = img_atten_pos3.shape[2:]
        img_atten_pos1 = F.interpolate(img_atten_pos1, size=final_sz, mode='nearest')
        img_atten_pos2 = F.interpolate(img_atten_pos2, size=final_sz, mode='nearest')
        img_atten_chn1 = F.interpolate(img_atten_chn1, size=final_sz, mode='nearest')
        img_atten_chn2 = F.interpolate(img_atten_chn2, size=final_sz, mode='nearest')
        img_atten = self.v_atten_fusion(
            torch.cat(
                (
                    img_atten_pos1,
                    img_atten_pos2,
                    img_atten_pos3,
                    img_atten_chn1,
                    img_atten_chn2,
                    img_atten_chn3,
                ),
                dim=1,
            )
        ).squeeze()

        # fuse pc attention
        final_sz = pc_atten_pos3.shape[2:]
        pc_atten_pos1 = F.interpolate(pc_atten_pos1, size=final_sz, mode='nearest')
        pc_atten_pos2 = F.interpolate(pc_atten_pos2, size=final_sz, mode='nearest')
        pc_atten_chn1 = F.interpolate(pc_atten_chn1, size=final_sz, mode='nearest')
        pc_atten_chn2 = F.interpolate(pc_atten_chn2, size=final_sz, mode='nearest')
        pc_atten = self.l_atten_fusion(
            torch.cat(
                (
                    pc_atten_pos1,
                    pc_atten_pos2,
                    pc_atten_pos3,
                    pc_atten_chn1,
                    pc_atten_chn2,
                    pc_atten_chn3,
                ),
                dim=1,
            )
        ).squeeze()

        # fuse img & pc attention, then use it to weight features
        img_pc_atten = torch.cat((img_atten, pc_atten), dim=1)
        img_pc_atten = self.vl_atten_fusion(img_pc_atten).unsqueeze(2)  # N*2*1
        #
        img_out = F.adaptive_avg_pool2d(img_fea_detach, (1, 1)).squeeze()
        pc_out = F.adaptive_avg_pool3d(pc_fea_detach, (1, 1, 1)).squeeze()
        img_out = img_out.unsqueeze(1)
        pc_out = pc_out.unsqueeze(1)
        img_pc_out = torch.cat((img_out, pc_out), dim=1)  # N*2*C

        concat = torch.reshape(
            torch.mul(img_pc_atten, img_pc_out), (img_pc_out.size(0), -1)
        )

        return concat

    
def aavislidarnet3(fusion_method, Lp: int):
    """fusion method: can be one of "cat", "linear" or "mlp".
    Lp: the norm p
    """
    return AAVisLidarNet3(fusionBlock(fusion_method), Lp)
