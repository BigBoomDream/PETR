import torch

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    将已经经过 Sigmoid 处理的值还原回其原始的、未压缩的范围。
    """
    x = x.clamp(min=eps, max=1 - eps)  # 避免数值不稳定
    return torch.log(x / (1 - x))

# 示例：生成原始输入（logits）
x = torch.randn(3)
print("Original logits:", x)

# 正向：应用 sigmoid 压缩到 [0, 1]
y = torch.sigmoid(x)
print("After sigmoid (probabilities):", y)

# 反向：使用 inverse_sigmoid 还原回 logit 空间
z = inverse_sigmoid(y)
print("Recovered logits:", z)

# 计算误差
difference = (x - z).abs().mean()
print("Mean absolute difference:", difference.item())