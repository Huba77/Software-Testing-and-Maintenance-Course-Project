
import torch

def donut_loss(recon_x, x, mu, logvar):

    recon_loss = ((x - recon_x) ** 2).mean()

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    loss = recon_loss + 0.01 * kl_loss

    return loss, recon_loss, kl_loss


# ========== 成员B新增：M-ELBO Loss with mask ==========

def donut_loss_melbo(recon_x, x, mu, logvar, mask=None, beta=0.01):
    """
    Modified ELBO (M-ELBO) Loss with masking.
    
    论文核心思想：
    - 用 mask α_w 屏蔽异常/缺失点，使它们不参与重构损失计算
    - 用 β 缩放 KL 散度项
    
    mask=1 表示正常点（参与计算），mask=0 表示异常/缺失点（被屏蔽）。
    
    Args:
        recon_x: 重构输出
        x: 原始输入
        mu: 编码器输出的均值
        logvar: 编码器输出的对数方差
        mask: 掩码，形状与x相同，1=正常点，0=异常/缺失点。
              为None时退化为标准ELBO。
        beta: KL项的权重系数
    
    Returns:
        loss, recon_loss, kl_loss
    """
    if mask is None:
        # 无mask时退化为标准ELBO
        recon_loss = ((x - recon_x) ** 2).mean()
    else:
        # M-ELBO: 只计算正常点的重构损失
        squared_error = (x - recon_x) ** 2
        masked_error = squared_error * mask
        # 除以mask之和进行归一化（避免mask全0时除0）
        recon_loss = masked_error.sum() / (mask.sum() + 1e-10)
    
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss


def create_missing_mask(x, missing_rate=0.01):
    """
    创建随机缺失掩码（用于训练时的M-ELBO）。
    
    Args:
        x: 输入张量
        missing_rate: 缺失比例
    
    Returns:
        mask: 1=正常点，0=缺失点
    """
    mask = (torch.rand_like(x) >= missing_rate).float()
    return mask


def create_abnormal_mask_from_recon_error(x, recon_x, percentile=95):
    """
    基于重构误差动态创建异常掩码。
    
    思路：重构误差大的点可能是异常点，在M-ELBO中屏蔽它们。
    
    Args:
        x: 原始输入
        recon_x: 重构输出
        percentile: 误差百分位数阈值，超过该阈值的点视为异常
    
    Returns:
        mask: 1=正常点，0=异常点
    """
    recon_error = (x - recon_x).abs()
    threshold = torch.quantile(recon_error, percentile / 100.0)
    mask = (recon_error <= threshold).float()
    return mask
