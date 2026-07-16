import numpy as np

def batch_norm_forward(z, gamma, beta, eps=1e-5):
    """
    批归一化前向传播
    z: 输入，形状 (N, D)，N为batch大小，D为特征维度
    gamma, beta: 可学习参数，形状 (D,)
    eps: 数值稳定性小量
    返回:
        out: 归一化后的输出，形状 (N, D)
        cache: 保存的中间变量，用于反向传播
    """
    mu = np.mean(z, axis=0, keepdims=True)          # 均值 (1, D)
    var = np.var(z, axis=0, keepdims=True)          # 方差 (1, D)
    z_hat = (z - mu) / np.sqrt(var + eps)           # 标准化 (N, D)
    out = gamma * z_hat + beta                      # 缩放平移 (N, D)
    cache = (z, mu, var, z_hat, gamma, beta, eps)
    return out, cache

def batch_norm_backward(dout, cache):
    """
    批归一化反向传播
    dout: 上游梯度，形状 (N, D)
    cache: 前向传播保存的中间变量
    返回:
        dz: 输入梯度，形状 (N, D)
        dgamma: gamma梯度，形状 (D,)
        dbeta: beta梯度，形状 (D,)
    """
    z, mu, var, z_hat, gamma, beta, eps = cache
    N = z.shape[0]
    # 梯度计算
    dgamma = np.sum(dout * z_hat, axis=0)                # (D,)
    dbeta = np.sum(dout, axis=0)                        # (D,)
    dz_hat = dout * gamma                               # (N, D)
    # dvar: 对方差的梯度
    dvar = np.sum(dz_hat * (z - mu) * (-0.5) * (var + eps)**(-1.5), axis=0)  # (D,)
    # dmu: 对均值的梯度
    dmu = np.sum(dz_hat * (-1) / np.sqrt(var + eps), axis=0) + dvar * np.mean(-2 * (z - mu), axis=0)  # (D,)
    # dz: 对输入的梯度
    dz = dz_hat / np.sqrt(var + eps) + dvar * 2 * (z - mu) / N + dmu / N      # (N, D)
    return dz, dgamma, dbeta

# ========== 测试代码 ==========
if __name__ == "__main__":
    np.random.seed(42)               # 固定随机种子，结果可复现
    N, D = 4, 3                      # batch大小4，特征维度3
    z = np.random.randn(N, D)        # 输入
    gamma = np.random.randn(D)       # 尺度参数
    beta = np.random.randn(D)        # 偏移参数

    # 前向传播
    out, cache = batch_norm_forward(z, gamma, beta)

    # 模拟上游梯度（例如来自损失函数的梯度）
    dout = np.random.randn(N, D)

    # 反向传播，计算梯度
    dz, dgamma, dbeta = batch_norm_backward(dout, cache)

    print("=== 前向输出 ===")
    print("out:\n", out)
    print("\n=== 反向梯度 ===")
    print("dz (部分):\n", dz[:2, :2])   # 只显示前2行前2列
    print("dgamma:", dgamma)
    print("dbeta:", dbeta)

    # ---------- 数值梯度检查（验证反向传播是否正确）----------
    print("\n=== 数值梯度检查 ===")
    eps_num = 1e-5                     # 用于数值微分的扰动
    # 检查 dgamma
    grad_num_gamma = np.zeros_like(gamma)
    for i in range(D):
        gamma_plus = gamma.copy()
        gamma_plus[i] += eps_num
        out_plus, _ = batch_norm_forward(z, gamma_plus, beta)
        gamma_minus = gamma.copy()
        gamma_minus[i] -= eps_num
        out_minus, _ = batch_norm_forward(z, gamma_minus, beta)
        # 假设损失函数 L = sum(out * dout) 的近似（实际上游梯度已给定，这里模拟）
        # 更通用的做法是使用一个假损失函数，例如 L = np.sum(out * dout)（因为dout = ∂L/∂out）
        # 这里我们直接用给定的 dout 计算数值梯度： ∂L/∂gamma ≈ (L(gamma+eps)-L(gamma-eps))/(2*eps)
        # 其中 L = np.sum(out * dout)（因为 dout 是上游梯度，所以 L 可以视为损失函数）
        L_plus = np.sum(out_plus * dout)
        L_minus = np.sum(out_minus * dout)
        grad_num_gamma[i] = (L_plus - L_minus) / (2 * eps_num)
    print("dgamma 数值梯度:", grad_num_gamma)
    print("dgamma 解析梯度:", dgamma)
    print("dgamma 相对误差:", np.linalg.norm(grad_num_gamma - dgamma) / (np.linalg.norm(grad_num_gamma) + 1e-8))

    # 检查 dbeta
    grad_num_beta = np.zeros_like(beta)
    for i in range(D):
        beta_plus = beta.copy()
        beta_plus[i] += eps_num
        out_plus, _ = batch_norm_forward(z, gamma, beta_plus)
        beta_minus = beta.copy()
        beta_minus[i] -= eps_num
        out_minus, _ = batch_norm_forward(z, gamma, beta_minus)
        L_plus = np.sum(out_plus * dout)
        L_minus = np.sum(out_minus * dout)
        grad_num_beta[i] = (L_plus - L_minus) / (2 * eps_num)
    print("\ndbeta 数值梯度:", grad_num_beta)
    print("dbeta 解析梯度:", dbeta)
    print("dbeta 相对误差:", np.linalg.norm(grad_num_beta - dbeta) / (np.linalg.norm(grad_num_beta) + 1e-8))

    # 检查 dz (需要利用损失函数 L = np.sum(out * dout)，则 ∂L/∂z 应等于解析 dz)
    grad_num_z = np.zeros_like(z)
    for i in range(N):
        for j in range(D):
            z_plus = z.copy()
            z_plus[i, j] += eps_num
            out_plus, _ = batch_norm_forward(z_plus, gamma, beta)
            z_minus = z.copy()
            z_minus[i, j] -= eps_num
            out_minus, _ = batch_norm_forward(z_minus, gamma, beta)
            L_plus = np.sum(out_plus * dout)
            L_minus = np.sum(out_minus * dout)
            grad_num_z[i, j] = (L_plus - L_minus) / (2 * eps_num)
    print("\ndz 数值梯度（部分）:\n", grad_num_z[:2, :2])
    print("dz 解析梯度（部分）:\n", dz[:2, :2])
    print("dz 相对误差:", np.linalg.norm(grad_num_z - dz) / (np.linalg.norm(grad_num_z) + 1e-8))