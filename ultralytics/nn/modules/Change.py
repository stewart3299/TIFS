import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureInference(nn.Module):
    """
    参考 TCDG-Net 项目中的 GetMap.py 实现 
    利用 Sobel 算子提取双模态梯度，并合成共识纹理图 M
    对应论文公式(3): M = 1 - (G1 * G2 + (1 - G1) * (1 - G2))
    """
    def __init__(self, *args):
        super(TextureInference, self).__init__()
        # 定义 Sobel 算子 (水平 H 和 垂直 V) 
        kernel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_h', kernel_h)
        self.register_buffer('sobel_v', kernel_v)

    def get_gradient_amplitude(self, x):
        """
        计算梯度幅值并归一化
        Args:
            x: 输入图像/特征 [B, C, H, W]
        Returns:
            归一化的梯度幅值 [B, 1, H, W]
        """
        # 如果输入是多通道，先转换为灰度
        if x.shape[1] >= 3:
            # RGB转灰度: 0.299R + 0.587G + 0.114B
            gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        else:
            gray = x
        
        # 计算水平和垂直梯度
        grad_h = F.conv2d(gray, self.sobel_h, padding=1)
        grad_v = F.conv2d(gray, self.sobel_v, padding=1)
        
        # 计算梯度幅值（使用L1范数）
        grad_amp = (torch.abs(grad_h) + torch.abs(grad_v)) / 2
        
        # 归一化到 [0, 1]
        grad_min = grad_amp.min()
        grad_max = grad_amp.max()
        grad_amp_norm = (grad_amp - grad_min) / (grad_max - grad_min + 1e-8)
        
        return grad_amp_norm

    def forward(self, img_rgb, img_ir):
        """
        计算纹理共识图 M
        Args:
            img_rgb: RGB图像/特征 [B, C, H, W] (C通常为3)
            img_ir: 红外图像/特征 [B, C, H, W] (C通常为3)
        Returns:
            m_map: 纹理共识图 [B, 1, H, W]
        """
        # print(f'输入RGB特征形状: {img_rgb.shape}, 输入IR特征形状: {img_ir.shape}')
        # 1. 分别获取 RGB 和 IR 的归一化梯度幅值 G1, G2
        g1 = self.get_gradient_amplitude(img_rgb)
        g2 = self.get_gradient_amplitude(img_ir)
        
        # 2. 按照论文公式 (3) 计算纹理图 M
        # 公式: M = 1 - (G1 * G2 + (1 - G1) * (1 - G2))
        # 物理意义：强调各模态独特的边缘，抑制共有的冗余信息
        m_map = 1 - (g1 * g2 + (1 - g1) * (1 - g2))
        # print(f'输出纹理图形状: {m_map.shape}')

        
        return m_map
    