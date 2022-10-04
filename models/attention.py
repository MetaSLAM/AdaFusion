import torch
import torch.nn as nn
import torch.nn.functional as F


class M_PosAttention2d(nn.Module):
    """Multi-head Position Attention 2D Module.
       Attention augmentation convolution, pytorch implementation of paper
    [1] Bello I, Zoph B, Vaswani A, et al. Attention augmented convolutional
        networks[C]//ICCV. 2019: 3286-3295.

        This class only contains the pure attention part for 2D inputs. The 
    combination of convolution should be done elsewhere outside this class.
        Currently no [relative] position encoding is available.

        Attention output channels: dv
    Args:
        dk, dv: Channels for K and V
        Nh: The number of heads
        dkh, dvh: Channels for K and V per head
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size,
        dk: int,
        dv: int,
        Nh: int,
        stride: int = 1,
        singleHead=False,
    ):
        """Note that out_channels = dv."""
        super(M_PosAttention2d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        # check parameters
        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."
        #
        self.dkh = self.dk // self.Nh
        self.dvh = self.dv // self.Nh

        # W_q, W_k, W_v as a whole matrix
        self.qkv_conv = nn.Conv2d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # W_O matrix
        self.attn_out = (
            nn.Identity()
            if singleHead
            else nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)
        )

    def forward(self, x):
        """Input x, shape (N, in_channels, H_ori, W_ori)"""

        # [Q,K,V] matrix, shape (N,2*dk+dv,H,W)
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()

        # shape (N, [dk, dk, dv], H, W)
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (self.dkh ** -0.5)  # the scale 1/sqrt(dkh) is multiplied to Q

        # split to multi-head, shape (N, Nh, dkh, H, W)
        q = torch.reshape(q, (N, self.Nh, self.dkh, H, W))

        # flatten Q, K or V. Combine (H,W) into (H*W,) shape
        # shape (N, Nh, dkh, H*W)
        flat_q = torch.reshape(q, (N, self.Nh, self.dkh, H * W))
        flat_k = torch.reshape(k, (N, self.Nh, self.dkh, H * W))
        flat_v = torch.reshape(v, (N, self.Nh, self.dvh, H * W))

        # logits = QK^T / sqrt(dkh), shape (N, Nh, H*W, H*W)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=-1)  # in [0, 1]

        # shape (N, Nh, H*W, dvh) -> (N, Nh, dvh, H*W)
        Oh = torch.matmul(weights, flat_v.transpose(2, 3))
        Oh = Oh.transpose(2, 3)

        # combine_heads O_all=[O1, O2, ... O_Nh], shape (N, dv, H, W)
        # attention out = O_all * W_O, shape (N, dv, H, W)
        O_all = torch.reshape(Oh, (N, self.dv, H, W))
        attn_out = self.attn_out(O_all)

        return attn_out


class M_PosAttention3d(nn.Module):
    """Multi-head Position Attention 3D Module.
       Attention augmentation convolution, pytorch implementation of paper
    [1] Bello I, Zoph B, Vaswani A, et al. Attention augmented convolutional
        networks[C]//ICCV. 2019: 3286-3295.

        This class only contains the pure attention part for 3D inputs. The 
    combination of convolution should be done elsewhere outside this class.
        Currently no [relative] position encoding is available.

        Attention output channels: dv
    Args:
        dk, dv: Channels for K and V
        Nh: The number of heads
        dkh, dvh: Channels for K and V per head
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size,
        dk: int,
        dv: int,
        Nh: int,
        stride: int = 1,
        singleHead=False,
    ):
        """Note that out_channels = dv."""
        super(M_PosAttention3d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        # check parameters
        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."
        #
        self.dkh = self.dk // self.Nh
        self.dvh = self.dv // self.Nh

        # W_q, W_k, W_v as a whole matrix
        self.qkv_conv = nn.Conv3d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # W_O matrix
        self.attn_out = (
            nn.Identity()
            if singleHead
            else nn.Conv3d(self.dv, self.dv, kernel_size=1, stride=1)
        )

    def forward(self, x):
        """Input x, shape (N, in_channels, X_ori, Y_ori, Z_ori)"""

        # [Q,K,V] matrix, shape (N, 2*dk+dv, X, Y, Z)
        qkv = self.qkv_conv(x)
        N, _, X, Y, Z = qkv.size()

        # shape (N, [dk, dk, dv], X, Y, Z)
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (self.dkh ** -0.5)  # the scale 1/sqrt(dkh) is multiplied to Q

        # split to multi-head, shape (N, Nh, dkh, X, Y, Z)
        q = torch.reshape(q, (N, self.Nh, self.dkh, X, Y, Z))

        # flatten Q, K or V. Combine (X,Y,Z) into (X*Y*Z,) shape
        # shape (N, Nh, dkh, X*Y*Z)
        flat_q = torch.reshape(q, (N, self.Nh, self.dkh, X * Y * Z))
        flat_k = torch.reshape(k, (N, self.Nh, self.dkh, X * Y * Z))
        flat_v = torch.reshape(v, (N, self.Nh, self.dvh, X * Y * Z))

        # logits = QK^T / sqrt(dkh), shape (N, Nh, X*Y*Z, X*Y*Z)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=-1)  # in [0, 1]

        # shape (N, Nh, X*Y*Z, dvh) -> (N, Nh, dvh, X*Y*Z)
        Oh = torch.matmul(weights, flat_v.transpose(2, 3))
        Oh = Oh.transpose(2, 3)

        # combine_heads O_all=[O1, O2, ... O_Nh], shape (N, dv, X, Y, Z)
        # attention out = O_all * W_O, shape (N, dv, X, Y, Z)
        O_all = torch.reshape(Oh, (N, self.dv, X, Y, Z))
        attn_out = self.attn_out(O_all)

        return attn_out


class S_ChnAttention2d(nn.Module):
    """Single-head Channel Attention 2D Module.
       We implement the single head channel self-attention from paper
    [2] Fu, Jun, et al. "Dual attention network for scene segmentation." CVPR. 2019.

        This class only contains the pure attention part for 2D inputs. The 
    combination of convolution should be done elsewhere outside this class.
    """

    def __init__(self, in_chn: int, out_chn: int):
        """Args:
            in_chn: the input channal number
            out_chn: the desired output channal number"""
        super(S_ChnAttention2d, self).__init__()
        self.bottleneck = nn.Conv2d(in_chn, out_chn, kernel_size=1)

    def forward(self, x):
        """Input x, shape (N, C, H, W)"""

        N, C, H, W = x.size()  # shape (N, C, H, W)

        # combine H and W
        x_HW = torch.reshape(x, (N, C, H * W))  # shape (N, C, H*W)

        # A * A^T, is a symmetric matrix
        logits = torch.matmul(x_HW, x_HW.transpose(1, 2))  # shape (N, C, C)
        weights = F.softmax(logits, dim=-1)  # row in [0, 1], shape (N, C, C)

        # attention output
        attn_out = torch.matmul(weights, x_HW)  # shape (N, C, H*W)
        attn_out = torch.reshape(attn_out, (N, C, H, W))
        attn_out = self.bottleneck(attn_out)

        return attn_out


class S_ChnAttention3d(nn.Module):
    """Single-head Channel Attention 3D Module.
       We implement the single head channel self-attention from paper
    [2] Fu, Jun, et al. "Dual attention network for scene segmentation." CVPR. 2019.

        This class only contains the pure attention part for 3D inputs. The 
    combination of convolution should be done elsewhere outside this class.
    """

    def __init__(self, in_chn: int, out_chn: int):
        """Args:
            in_chn: the input channal number
            out_chn: the desired output channal number"""
        super(S_ChnAttention3d, self).__init__()
        self.bottleneck = nn.Conv3d(in_chn, out_chn, kernel_size=1)

    def forward(self, x):
        """Input x, shape (N, C, X, Y, Z)"""

        N, C, X, Y, Z = x.size()  # shape (N, C, X, Y, Z)

        # combine X, Y and Z
        x_XYZ = torch.reshape(x, (N, C, X * Y * Z))  # shape (N, C, X*Y*Z)

        # A * A^T, is a symmetric matrix
        logits = torch.matmul(x_XYZ, x_XYZ.transpose(1, 2))  # shape (N, C, C)
        weights = F.softmax(logits, dim=-1)  # row in [0, 1], shape (N, C, C)

        # attention output
        attn_out = torch.matmul(weights, x_XYZ)  # shape (N, C, X*Y*Z)
        attn_out = torch.reshape(attn_out, (N, C, X, Y, Z))
        attn_out = self.bottleneck(attn_out)

        return attn_out
