import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from dataclasses import dataclass, field
from typing import List, Tuple
import time

# ==============================================================================
# 模块1：基础工具（矩形FFT修正版）
# ==============================================================================
PI = np.pi
EPS = 1e-12

def fft2c_rect(x: np.ndarray) -> np.ndarray:
    """矩形网格中心化FFT2：零频居中"""
    return np.fft.fftshift(np.fft.fft2(x))

def ifft2c_rect(x: np.ndarray) -> np.ndarray:
    """矩形网格中心化IFFT2"""
    return np.fft.ifft2(np.fft.ifftshift(x))

def crop_kernel_center_rect(kernel: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """矩形空域核中心裁剪（支持非对称）"""
    H, W = kernel.shape
    c_h, c_w = H // 2, W // 2
    hs_h, hs_w = crop_h // 2, crop_w // 2
    return kernel[c_h-hs_h:c_h+hs_h, c_w-hs_w:c_w+hs_w]

def pad_kernel_to_size_rect(kernel: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """矩形核补零到目标尺寸（中心放置）"""
    H, W = kernel.shape
    padded = np.zeros((target_h, target_w), dtype=kernel.dtype)
    c_h, c_w = target_h // 2, target_w // 2
    hs_h, hs_w = H // 2, W // 2
    padded[c_h-hs_h:c_h+hs_h, c_w-hs_w:c_w+hs_w] = kernel
    return padded

def min_nyquist_kernel_size_rect(lambda_nm: float, NA: float, dx_nm: float, dy_nm: float) -> Tuple[int, int]:
    """矩形网格奈奎斯特最小尺寸（奇数强制）"""
    fc = NA / lambda_nm
    min_px_h = np.ceil(1.0 / (dy_nm * 2 * fc))
    min_px_w = np.ceil(1.0 / (dx_nm * 2 * fc))
    
    # 强制奇数
    min_h = int(min_px_h) if min_px_h % 2 == 1 else int(min_px_h + 1)
    min_w = int(min_px_w) if min_px_w % 2 == 1 else int(min_px_w + 1)
    return min_h, min_w

def kernel_energy_ratio_rect(full_kernel: np.ndarray, cropped: np.ndarray) -> float:
    """矩形缩核能量保留率"""
    E_full = np.sum(np.abs(full_kernel)**2)
    E_crop = np.sum(np.abs(cropped)**2)
    return E_crop / (E_full + EPS)

# ==============================================================================
# 模块2：光学系统参数（矩形专用）
# ==============================================================================
@dataclass
class LithoOpticsRect:
    # 物理参数
    lambda_nm: float = 193.0          # ArF激光波长 [nm]
    NA: float = 0.85                  # 数值孔径
    dx_nm: float = 1.0                # x方向采样 [nm]
    dy_nm: float = 1.0                # y方向采样 [nm]
    N_x: int = 2048                   # x方向网格数
    N_y: int = 1024                   # y方向网格数（非对称）
    
    # 传统光源专用参数
    src_type: str = field(default="conventional", init=False)  # 固定为传统光源
    sigma_c: float = 0.5              # 相干因子
    N_source: int = 200               # 光源采样点数
    
    # SOCS参数
    n_socs: int = 32                  # 保留阶数
    kernel_crop_h: int = 64           # 核裁剪高度
    kernel_crop_w: int = 64           # 核裁剪宽度
    
    # 掩模图形
    line_width_nm: float = 100.0      # 线宽 [nm]
    pitch_nm: float = 400.0           # 周期 [nm]
    
    # 像差
    defocus_z4: float = 0.0           # 离焦像差
    
    def __post_init__(self):
        """参数校验"""
        assert self.N_x % 2 == 0 and self.N_y % 2 == 0, "网格尺寸必须为偶数"
        assert self.n_socs <= self.N_source, "SOCS阶数不能超过光源采样数"
        assert self.kernel_crop_h <= self.N_y and self.kernel_crop_w <= self.N_x, "裁剪尺寸不能超过网格"
        print(f"矩形网格配置: {self.N_y}×{self.N_x} (y×x)")

# ==============================================================================
# 模块3：矩形频域网格与光瞳
# ==============================================================================
def freq_grid_rect(opt: LithoOpticsRect) -> Tuple[np.ndarray, np.ndarray]:
    """矩形频域坐标网格 [1/nm]"""
    fx = np.fft.fftfreq(opt.N_x, d=opt.dx_nm)
    fy = np.fft.fftfreq(opt.N_y, d=opt.dy_nm)
    return np.meshgrid(fx, fy)

def zernike_z4_defocus(rho: np.ndarray) -> np.ndarray:
    """Z4离焦像差多项式: 2ρ² - 1"""
    return 2 * rho**2 - 1

def pupil_rect(fx: np.ndarray, fy: np.ndarray, opt: LithoOpticsRect) -> np.ndarray:
    """矩形光瞳函数: 振幅截止 + 离焦相位"""
    fc = opt.NA / opt.lambda_nm
    f_rho = np.hypot(fx, fy)
    
    P = np.zeros_like(f_rho, dtype=np.complex128)
    valid = f_rho <= fc
    P[valid] = 1.0 + 0j
    
    if abs(opt.defocus_z4) > EPS:
        rho_norm = f_rho[valid] / fc
        phase = PI * opt.defocus_z4 * zernike_z4_defocus(rho_norm)
        P[valid] *= np.exp(1j * phase)
    
    return P

def shifted_pupil_rect(
    fx: np.ndarray, 
    fy: np.ndarray, 
    f_shift: np.ndarray, 
    opt: LithoOpticsRect
) -> np.ndarray:
    """矩形网格偏移光瞳（独立计算，避免复用）"""
    fc = opt.NA / opt.lambda_nm
    fx_shifted = fx - f_shift[0]
    fy_shifted = fy - f_shift[1]
    f_rho = np.hypot(fx_shifted, fy_shifted)
    
    P = np.zeros_like(f_rho, dtype=np.complex128)
    valid = f_rho <= fc
    P[valid] = 1.0 + 0j
    
    if abs(opt.defocus_z4) > EPS:
        rho_norm = f_rho[valid] / fc
        phase = PI * opt.defocus_z4 * zernike_z4_defocus(rho_norm)
        P[valid] *= np.exp(1j * phase)
    
    return P

# ==============================================================================
# 模块4：传统光源均匀采样（简化版）
# ==============================================================================
def sample_conventional_source(opt: LithoOpticsRect) -> Tuple[np.ndarray, np.ndarray]:
    """
    传统光源圆盘均匀面积采样（简化版）
    Returns: (fs_coords [N,2], weights [N])
    """
    fc = opt.NA / opt.lambda_nm
    Ns = opt.N_source
    
    # 圆盘面积均匀采样
    r_max = opt.sigma_c * fc
    # 半径平方均匀分布
    areas = np.linspace(0, r_max**2, int(np.sqrt(Ns)) + 1)
    r = np.sqrt(areas[1:])  # 排除r=0中心点
    # 角度均匀分布
    theta = np.linspace(0, 2*PI, Ns, endpoint=False)
    
    # 网格化并展平
    R, T = np.meshgrid(r, theta)
    fx_s = (R * np.cos(T)).ravel()
    fy_s = (R * np.sin(T)).ravel()
    
    # 截断到设定数量
    N_actual = min(Ns, len(fx_s))
    fx_s = fx_s[:N_actual]
    fy_s = fy_s[:N_actual]
    
    # 均匀权重（面积已归一化）
    weights = np.ones(N_actual) / N_actual
    
    return np.stack([fx_s, fy_s], axis=1), weights

# ==============================================================================
# 模块5：无TCC模式矩阵（矩形版）
# ==============================================================================
def build_mode_matrix_M_rect(opt: LithoOpticsRect, fx: np.ndarray, fy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造模式矩阵 M ∈ C^(Ny*Nx × N_source)
    矩形网格优化，无TCC，支持像差
    """
    Ny, Nx = opt.N_y, opt.N_x
    Ns = opt.N_source
    
    # 采样传统光源
    fs_coords, weights = sample_conventional_source(opt)
    
    M = np.zeros((Ny*Nx, Ns), dtype=np.complex128)
    print(f"\n构建模式矩阵 M: {Ny*Nx}×{Ns} = {M.nbytes/1e6:.1f} MB")
    
    for i in range(Ns):
        fx_s, fy_s = fs_coords[i]
        w = np.sqrt(weights[i] + EPS)
        
        # 双偏移光瞳（独立计算）
        P1 = shifted_pupil_rect(fx, fy, np.array([-fx_s/2, -fy_s/2]), opt)
        P2 = shifted_pupil_rect(fx, fy, np.array([+fx_s/2, +fy_s/2]), opt)
        
        # 频域模式向量
        vec_mode = (P1 * P2.conj()).ravel()
        M[:, i] = vec_mode * w
        
        if (i+1) % 50 == 0:
            print(f"  光源点 {i+1}/{Ns} 处理完成")
    
    return M

# ==============================================================================
# 模块6：SOCS分解（带能量校验）
# ==============================================================================
def socs_decompose_rect(M: np.ndarray, n_socs: int, energy_threshold: float = 0.9999) -> Tuple[np.ndarray, np.ndarray]:
    """
    经济SVD分解，自动校验能量占比
    """
    print(f"\n执行经济SVD分解...")
    start = time.time()
    
    U, sigma, _ = svd(M, full_matrices=False, overwrite_a=True)
    alpha = sigma[:n_socs]**2
    
    # 能量校验
    total_energy = np.sum(sigma**2)
    retained_energy = np.sum(alpha)
    energy_ratio = retained_energy / (total_energy + EPS)
    
    print(f"  保留能量: {retained_energy:.2e} / {total_energy:.2e} = {energy_ratio:.6f}")
    print(f"  目标阈值: {energy_threshold}")
    
    if energy_ratio < energy_threshold:
        print(f"  ⚠️ 警告：能量保留率不足，建议增加n_socs至{int(n_socs/energy_ratio)}")
    
    V = U[:, :n_socs]
    print(f"  SVD耗时: {time.time()-start:.2f}s")
    
    return alpha, V

# ==============================================================================
# 模块7：矩形空域核生成（带强制奈奎斯特）
# ==============================================================================
def build_socs_spatial_kernels_rect(
    opt: LithoOpticsRect,
    V: np.ndarray,
    alpha: np.ndarray
) -> List[np.ndarray]:
    """
    生成矩形空域核并执行单核能量校验
    """
    Ny, Nx = opt.N_y, opt.N_x
    n_socs = V.shape[1]
    
    # 强制奈奎斯特约束
    min_h, min_w = min_nyquist_kernel_size_rect(
        opt.lambda_nm, opt.NA, opt.dx_nm, opt.dy_nm
    )
    
    if opt.kernel_crop_h < min_h or opt.kernel_crop_w < min_w:
        print(f"⚠️ 裁剪尺寸({opt.kernel_crop_h}×{opt.kernel_crop_w})小于奈奎斯特最小({min_h}×{min_w})，已自动调整")
        crop_h, crop_w = max(opt.kernel_crop_h, min_h), max(opt.kernel_crop_w, min_w)
    else:
        crop_h, crop_w = opt.kernel_crop_h, opt.kernel_crop_w
    
    kernels = []
    energy_ratios = []
    
    print(f"\n生成空域核（裁剪={crop_h}×{crop_w}）...")
    
    for k in range(n_socs):
        # 频域 → 空域
        freq_k = V[:, k].reshape(Ny, Nx)
        spat_full = ifft2c_rect(freq_k) * np.sqrt(Ny*Nx)  # 能量守恒归一化
        
        # 矩形裁剪
        spat_crop = crop_kernel_center_rect(spat_full, crop_h, crop_w)
        
        # 单核能量校验
        ratio = kernel_energy_ratio_rect(spat_full, spat_crop)
        energy_ratios.append(ratio)
        
        kernels.append(spat_crop)
        
        if (k+1) % 10 == 0:
            print(f"  核 {k+1}/{n_socs} 能量保留率: {ratio:.6f}")
    
    # 统计校验
    avg_ratio = np.mean(energy_ratios)
    min_ratio = np.min(energy_ratios)
    print(f"\n缩核统计:")
    print(f"  平均能量保留率: {avg_ratio:.6f} (目标≥0.9999)")
    print(f"  最低能量保留率: {min_ratio:.6f}")
    
    return kernels

# ==============================================================================
# 模块8：矩形卷积成像
# ==============================================================================
def binary_line_mask_rect(opt: LithoOpticsRect) -> np.ndarray:
    """矩形网格二元线条掩模（沿x方向）"""
    x = (np.arange(opt.N_x) - opt.N_x//2) * opt.dx_nm
    y = (np.arange(opt.N_y) - opt.N_y//2) * opt.dy_nm
    xx, yy = np.meshgrid(x, y)
    
    # 沿x方向周期性线条
    mod = np.mod(xx + opt.pitch_nm/2, opt.pitch_nm)
    mask = (mod < opt.line_width_nm).astype(np.complex128)
    return mask

def fft_convolve_rect(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """矩形FFT卷积（核尺寸≤图像尺寸）"""
    Ny, Nx = img.shape
    ky, kx = kernel.shape
    
    # 核补零到图像尺寸
    kernel_padded = pad_kernel_to_size_rect(kernel, Ny, Nx)
    
    # FFT卷积
    img_freq = fft2c_rect(img)
    ker_freq = fft2c_rect(kernel_padded)
    result = ifft2c_rect(img_freq * ker_freq)
    
    return result

def aerial_image_socs_rect(
    opt: LithoOpticsRect,
    mask: np.ndarray,
    kernels: List[np.ndarray],
    alpha: np.ndarray
) -> np.ndarray:
    """矩形Hopkins SOCS空中图像合成"""
    I = np.zeros((opt.N_y, opt.N_x), dtype=np.float64)
    
    print(f"\n合成空中图像（{opt.n_socs}阶SOCS）...")
    start = time.time()
    
    for k, ker in enumerate(kernels):
        conv = fft_convolve_rect(mask, ker)
        I += alpha[k] * np.abs(conv)**2
        
        if (k+1) % 10 == 0:
            print(f"  阶次 {k+1}/{len(kernels)} 完成")
    
    I_norm = I / (np.max(I) + EPS)
    print(f"  成像耗时: {time.time()-start:.2f}s")
    
    return I_norm

# ==============================================================================
# 模块9：矩形可视化与质量报告
# ==============================================================================
def visualize_results_rect(
    opt: LithoOpticsRect,
    mask: np.ndarray,
    aerial: np.ndarray,
    kernels: List[np.ndarray],
    alpha: np.ndarray,
    save_path: str = "litho_rect_1024x2048.png"
):
    """矩形结果多维度可视化"""
    fig = plt.figure(figsize=(18, 10))
    
    # 1. 掩模
    ax1 = plt.subplot(2, 3, 1)
    plt.imshow(np.abs(mask), cmap="gray", origin="lower", aspect='auto')
    plt.title(f"掩模: {opt.N_y}×{opt.N_x} 像素")
    plt.colorbar()
    
    # 2. 空中图像
    ax2 = plt.subplot(2, 3, 2)
    im = plt.imshow(aerial, cmap="gray", origin="lower", aspect='auto', vmin=0, vmax=1)
    plt.title("空中图像 (SOCS合成)")
    plt.colorbar(im, label="归一化强度")
    
    # 3. x方向截面（沿线条周期）
    ax3 = plt.subplot(2, 3, 3)
    center_y = opt.N_y // 2
    profile_x = aerial[center_y, :]
    plt.plot(profile_x)
    plt.title(f"Y={center_y}截面 (对比度={profile_x.max()-profile_x.min():.3f})")
    plt.xlabel("X像素")
    plt.ylabel("强度")
    plt.grid(True)
    
    # 4. y方向截面（垂直线条）
    ax4 = plt.subplot(2, 3, 4)
    center_x = opt.N_x // 2
    profile_y = aerial[:, center_x]
    plt.plot(profile_y)
    plt.title(f"X={center_x}截面")
    plt.xlabel("Y像素")
    plt.ylabel("强度")
    plt.grid(True)
    
    # 5. 前3阶核可视化
    for i in range(min(3, len(kernels))):
        ax = plt.subplot(2, 3, 5+i)
        ker = np.abs(kernels[i])
        plt.imshow(ker, cmap="hot", origin="lower")
        plt.title(f"核 #{i+1} (α={alpha[i]:.2e})")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    # 质量报告
    print("\n" + "="*60)
    print("工业级矩形仿真质量报告")
    print("="*60)
    print(f"网格尺寸: {opt.N_y}×{opt.N_x} (y×x)")
    print(f"空域采样: dx={opt.dx_nm} nm, dy={opt.dy_nm} nm")
    print(f"光学系统: λ={opt.lambda_nm}nm, NA={opt.NA:.2f}, σ={opt.sigma_c:.2f}")
    print(f"SOCS阶数: {opt.n_socs}")
    print(f"核尺寸: {opt.kernel_crop_h}×{opt.kernel_crop_w} (y×x)")
    
    # 对比度评估
    center_y = opt.N_y // 2
    profile_x = aerial[center_y, :]
    contrast = profile_x.max() - profile_x.min()
    print(f"图像对比度: {contrast:.4f}")
    
    # 内存与性能
    memory_peak = (opt.N_y * opt.N_x * opt.n_socs * 16) / 1e9  # GB
    print(f"峰值内存估算: ~{memory_peak:.2f} GB")

# ==============================================================================
# 主流程：矩形专用执行管线
# ==============================================================================
def run_litho_simulation_rect(opt: LithoOpticsRect) -> dict:
    """矩形网格完整仿真管线"""
    print("\n" + "="*60)
    print("启动工业级矩形Hopkins SOCS光刻仿真")
    print("="*60)
    
    start_total = time.time()
    
    # 1. 频域网格
    print("\n[1/7] 生成矩形频域网格...")
    fx, fy = freq_grid_rect(opt)
    
    # 2. 构建模式矩阵M
    print("\n[2/7] 构建模式矩阵M（无TCC）...")
    M = build_mode_matrix_M_rect(opt, fx, fy)
    
    # 3. SOCS分解
    print("\n[3/7] SOCS分解（经济SVD）...")
    alpha, V = socs_decompose_rect(M, opt.n_socs)
    
    # 4. 生成空域核
    print("\n[4/7] 生成矩形空域核并执行缩核...")
    kernels = build_socs_spatial_kernels_rect(opt, V, alpha)
    
    # 5. 掩模
    print("\n[5/7] 生成二元矩形掩模...")
    mask = binary_line_mask_rect(opt)
    
    # 6. 成像
    print("\n[6/7] 合成矩形空中图像...")
    aerial = aerial_image_socs_rect(opt, mask, kernels, alpha)
    
    # 7. 可视化
    print("\n[7/7] 矩形结果可视化与质量校验...")
    visualize_results_rect(opt, mask, aerial, kernels, alpha)
    
    print(f"\n✅ 仿真完成！总耗时: {time.time()-start_total:.2f}s")
    
    return {
        "opt": opt,
        "mask": mask,
        "aerial": aerial,
        "kernels": kernels,
        "alpha": alpha,
        "V": V,
        "M": M
    }

# ==============================================================================
# 配置与执行
# ==============================================================================
if __name__ == "__main__":
    # ============== 1024×2048专用配置 ==============
    opt_1024x2048 = LithoOpticsRect(
        lambda_nm=193.0,
        NA=0.85,
        dx_nm=1.0,
        dy_nm=1.0,
        N_x=2048,                    # 宽方向（x）
        N_y=1024,                    # 高方向（y）
        sigma_c=0.5,
        N_source=300,                # 增加采样精度
        n_socs=32,
        kernel_crop_h=64,            # y方向裁剪
        kernel_crop_w=64,            # x方向裁剪
        line_width_nm=100.0,
        pitch_nm=400.0,
        defocus_z4=0.0
    )
    
    # 执行仿真
    results = run_litho_simulation_rect(opt_1024x2048)

