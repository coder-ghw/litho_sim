import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ===================== 全局配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
PI = torch.pi
EPS = 1e-12

# ===================== 可微工具函数（适配非正方形） =====================
def fft2c(x: torch.Tensor) -> torch.Tensor:
    """中心化FFT2（支持非正方形）"""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))

def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """中心化IFFT2（支持非正方形）"""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))

def crop_kernel_center(kernel: torch.Tensor, crop_size_h: int, crop_size_w: int) -> torch.Tensor:
    """适配非正方形核裁剪：h=高度, w=宽度"""
    H, W = kernel.shape
    c_h, c_w = H // 2, W // 2
    hs_h, hs_w = crop_size_h // 2, crop_size_w // 2
    return kernel[c_h-hs_h:c_h+hs_h, c_w-hs_w:c_w+hs_w]

def min_nyquist_kernel_size(lambda_: float, NA: float, dx: float) -> Tuple[int, int]:
    """返回 (H,W) 最小无损核尺寸（非正方形也强制奇数）"""
    fc = NA / lambda_
    min_px = np.ceil(1.0 / (dx * 2 * fc)).item()
    min_px = int(min_px) if min_px % 2 == 1 else int(min_px + 1)
    return (min_px, min_px)  # 核仍用正方形，适配非正方图像

# ===================== 可微 Zernike 像差 =====================
def zernike_z4_defocus(fx: torch.Tensor, fy: torch.Tensor, NA: float, lambda_: float) -> torch.Tensor:
    fc = NA / lambda_
    f_norm = torch.hypot(fx, fy) / (fc + EPS)
    f_norm = torch.clamp(f_norm, 0.0, 1.0)
    return 2 * f_norm**2 - 1

# ===================== 核心：适配1024×2048的可微分成像器 =====================
class DiffSOCSImager_1024x2048(nn.Module):
    def __init__(
        self,
        lambda_: float = 193.0,
        NA: float = 0.85,
        dx: float = 1.0,
        H: int = 1024,    # 高度
        W: int = 2048,    # 宽度
        n_socs: int = 32,
        N_source: int = 200,
        crop_size: int = 64,  # 核仍用正方形64×64
        init_sigma_c: float = 0.5,
        init_defocus_z4: float = 0.0
    ):
        super().__init__()
        # 核心：非正方形网格参数
        self.H = H
        self.W = W
        self.lambda_ = lambda_
        self.NA = NA
        self.dx = dx
        self.n_socs = n_socs
        self.N_source = N_source
        self.crop_size = crop_size

        # 可优化参数（自动微分核心）
        self.sigma_c = nn.Parameter(torch.tensor(init_sigma_c, device=device))
        self.defocus_z4 = nn.Parameter(torch.tensor(init_defocus_z4, device=device))

        # 生成非正方形频域网格（关键修改！）
        self.fx, self.fy = self._make_freq_grid()
        # 奈奎斯特最小核尺寸
        self.min_h, self.min_w = min_nyquist_kernel_size(lambda_, NA, dx)
        self.final_crop_h = max(crop_size, self.min_h)
        self.final_crop_w = max(crop_size, self.min_w)

    def _make_freq_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成 1024×2048 非正方形频域网格"""
        # 高度H=1024，宽度W=2048
        fx = torch.fft.fftfreq(self.W, d=self.dx, device=device)  # 宽度对应fx
        fy = torch.fft.fftfreq(self.H, d=self.dx, device=device)  # 高度对应fy
        fx, fy = torch.meshgrid(fx, fy, indexing="xy")  # 输出 shape: (H, W)
        return fx, fy

    # ===================== 可微光瞳（适配非正方形） =====================
    def pupil(self) -> torch.Tensor:
        fc = self.NA / self.lambda_
        f = torch.hypot(self.fx, self.fy)
        P = torch.zeros((self.H, self.W), dtype=torch.complex128, device=device)
        P[f <= fc] = 1.0 + 0j

        # 可微像差相位
        phase = PI * self.defocus_z4 * zernike_z4_defocus(self.fx, self.fy, self.NA, self.lambda_)
        P[f <= fc] *= torch.exp(1j * phase[f <= fc])
        return P

    # ===================== 可微模式矩阵 M（适配1024×2048） =====================
    def build_mode_matrix_M(self) -> torch.Tensor:
        H, W = self.H, self.W
        Ns = self.N_source
        P = self.pupil()
        fs_pts = self._sample_source_conventional()
        fc = self.NA / self.lambda_

        # M 维度：H*W × N_source（1024*2048=2,097,152 × 200）
        M = torch.zeros((H*W, Ns), dtype=torch.complex128, device=device)
        for i in range(Ns):
            fx_s, fy_s = fs_pts[i]
            # 偏移频率（适配非正方形）
            f1x, f1y = self.fx + fx_s/2, self.fy + fy_s/2
            f2x, f2y = self.fx - fx_s/2, self.fy - fy_s/2

            f1, f2 = torch.hypot(f1x, f1y), torch.hypot(f2x, f2y)
            P1 = torch.zeros_like(P)
            P2 = torch.zeros_like(P)
            P1[f1 <= fc] = P[f1 <= fc]
            P2[f2 <= fc] = P[f2 <= fc]

            # 展平为向量（H*W）
            vec = (P1 * P2.conj()).flatten()
            M[:, i] = vec * torch.sqrt(torch.tensor(1.0, device=device))
        return M

    # ===================== 可微光源采样（保持正方形，不影响） =====================
    def _sample_source_conventional(self) -> torch.Tensor:
        fc = self.NA / self.lambda_
        r_max = torch.clamp(self.sigma_c, 0.01, 0.9) * fc
        n_r = int(np.sqrt(self.N_source * 0.3)) + 1
        n_theta = int(self.N_source / n_r) + 1

        r = torch.linspace(0, r_max, n_r, device=device)
        theta = torch.linspace(0, 2*PI, n_theta, device=device)
        rr, tt = torch.meshgrid(r, theta, indexing="xy")
        fx_s = rr * torch.cos(tt)
        fy_s = rr * torch.sin(tt)
        return torch.stack([fx_s.flatten(), fy_s.flatten()], dim=1)[:self.N_source]

    # ===================== 可微 SOCS 分解（适配H*W维度） =====================
    def socs_decompose(self, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        U, sigma, _ = torch.linalg.svd(M, full_matrices=False)
        alpha = sigma[:self.n_socs]**2
        V = U[:, :self.n_socs]  # shape: (H*W, n_socs)
        return alpha, V

    # ===================== 可微空域核（适配非正方形） =====================
    def build_spatial_kernels(self, V: torch.Tensor) -> List[torch.Tensor]:
        kernels = []
        for k in range(self.n_socs):
            # 向量转回 H×W 频域核
            freq_k = V[:, k].reshape(self.H, self.W)
            # IFFT 得到空域核
            spat = ifft2c(freq_k)
            # 中心裁剪（仍用正方形核，适配非正方图像）
            spat_crop = crop_kernel_center(spat, self.final_crop_h, self.final_crop_w)
            kernels.append(spat_crop)
        return kernels

    # ===================== 可微 FFT 卷积（适配1024×2048） =====================
    def conv(self, img: torch.Tensor, ker: torch.Tensor) -> torch.Tensor:
        H, W = self.H, self.W
        kH, kW = ker.shape
        # 核补零到 1024×2048
        pad = torch.zeros((H, W), dtype=torch.complex128, device=device)
        c_h, c_w = H//2, W//2
        hs_h, hs_w = kH//2, kW//2
        # 核心：边界检查，避免越界
        pad[
            max(0, c_h - hs_h):min(H, c_h + hs_h),
            max(0, c_w - hs_w):min(W, c_w + hs_w)
        ] = ker
        # FFT卷积
        return ifft2c(fft2c(img) * fft2c(pad))

    # ===================== 端到端可微前向（1024×2048） =====================
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # 1. 构建模式矩阵 M
        M = self.build_mode_matrix_M()
        # 2. SOCS分解
        alpha, V = self.socs_decompose(M)
        # 3. 生成空域核
        kernels = self.build_spatial_kernels(V)
        # 4. 成像
        I = torch.zeros((self.H, self.W), device=device)
        for a, k in zip(alpha, kernels):
            c = self.conv(mask, k)
            I += a * torch.abs(c)**2
        return I / torch.max(I)

# ===================== 生成1024×2048掩模 =====================
def make_line_mask_1024x2048(H: int=1024, W: int=2048, dx: float=1.0, lw_nm: float=100.0, pitch_nm: float=400.0) -> torch.Tensor:
    """生成 1024×2048 线条掩模"""
    # 高度方向x1: 1024px，宽度方向x2:2048px
    x1 = (torch.arange(H, device=device) - H//2) * dx
    x2 = (torch.arange(W, device=device) - W//2) * dx
    xx, yy = torch.meshgrid(x1, x2, indexing="xy")  # shape: (1024, 2048)
    # 沿宽度方向生成周期线条
    mod = torch.remainder(xx + pitch_nm/2, pitch_nm)
    return torch.where(mod < lw_nm, 1.0+0j, 0.0+0j).to(torch.complex128)

# ===================== 主流程：1024×2048 仿真 + 自动微分 =====================
if __name__ == "__main__":
    # 核心参数：1024×2048
    H = 1024
    W = 2048
    lambda_ = 193.0
    NA = 0.85
    dx = 1.0
    lw = 100.0
    pitch = 400.0

    # 1. 初始化可微分成像器（适配1024×2048）
    imager = DiffSOCSImager_1024x2048(
        lambda_=lambda_, NA=NA, dx=dx,
        H=H, W=W,          # 关键：设置非正方形尺寸
        n_socs=16,         # 速度/精度平衡，2048建议用32/64
        N_source=100,      # 光源采样数，200足够
        crop_size=64,      # 核尺寸，64×64加速
        init_sigma_c=0.2,  # 初始σ
        init_defocus_z4=0.0
    ).to(device)

    # 2. 生成1024×2048掩模
    mask = make_line_mask_1024x2048(H, W, dx, lw, pitch)
    print(f"掩模尺寸: {mask.shape} | 设备: {mask.device}")

    # 3. 生成目标图像（σ=0.5）
    with torch.no_grad():
        imager_gt = DiffSOCSImager_1024x2048(init_sigma_c=0.5, H=H, W=W).to(device)
        I_target = imager_gt(mask).detach()
    print(f"目标图像尺寸: {I_target.shape}")

    # 4. 优化器（适配大网格，调小学习率）
    optimizer = torch.optim.Adam(imager.parameters(), lr=5e-4)  # 1024×2048建议5e-4
    loss_fn = nn.MSELoss()

    # 5. 自动微分优化
    print("="*60)
    print(f"开始 1024×2048 自动微分优化 | 设备: {device}")
    print(f"初始σ: {imager.sigma_c.item():.3f} | 目标σ: 0.500")
    print("="*60)

    steps = 100  # 1024×2048建议100步足够
    loss_history = []
    sigma_history = []

    for step in range(steps):
        optimizer.zero_grad()
        I_pred = imager(mask)
        loss = loss_fn(I_pred, I_target)
        loss.backward()  # 自动微分，梯度回传
        optimizer.step()

        loss_history.append(loss.item())
        sigma_history.append(imager.sigma_c.item())

        if step % 10 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f} | σ: {imager.sigma_c.item():.3f}")

    # 6. 结果可视化（1024×2048图像）
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(loss_history)
    plt.title("Loss Curve (1024×2048)")
    plt.subplot(1,2,2)
    plt.plot(sigma_history, label="Optimized σ")
    plt.axhline(0.5, color='r', linestyle='--', label="Target σ=0.5")
    plt.legend()
    plt.show()

    # 最终图像对比
    with torch.no_grad():
        I_final = imager(mask)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(I_target.cpu().numpy(), cmap='gray')
    plt.title("Target (σ=0.5)")
    plt.subplot(1,3,2)
    plt.imshow(I_final.cpu().numpy(), cmap='gray')
    plt.title(f"Optimized (σ={imager.sigma_c.item():.3f})")
    plt.subplot(1,3,3)
    plt.imshow((I_target-I_final).cpu().numpy(), cmap='seismic')
    plt.title("Error")
    plt.tight_layout()
    plt.show()

    print(f"\n✅ 1024×2048 仿真完成！最终σ: {imager.sigma_c.item():.3f}")