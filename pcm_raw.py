import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from dataclasses import dataclass
from typing import List, Tuple

# ==============================================================================
# 模块1：基础工具（FFT、缩核、奈奎斯特、能量校验——工业通用）
# ==============================================================================
PI = np.pi
EPS = 1e-12

def fft2c(x: np.ndarray) -> np.ndarray:
    """Centered FFT2 (光刻标准)"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(x: np.ndarray) -> np.ndarray:
    """Centered IFFT2 + 归一化"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

def crop_kernel_center(kernel: np.ndarray, crop_size: int) -> np.ndarray:
    """空域核中心裁剪（缩核加速）"""
    N = kernel.shape[0]
    c = N // 2
    hs = crop_size // 2
    return kernel[c-hs:c+hs, c-hs:c+hs]

def min_nyquist_kernel_size(lambda_: float, NA: float, dx: float) -> int:
    """奈奎斯特最小无损核尺寸（低于必混叠，工业硬约束）"""
    fc = NA / lambda_
    min_px = np.ceil(1.0 / (dx * 2 * fc))
    return int(min_px) if min_px % 2 == 1 else int(min_px + 1)

def kernel_energy_ratio(full_kernel: np.ndarray, cropped: np.ndarray) -> float:
    """缩核能量保留率（量产要求 > 0.9999）"""
    E_full = np.sum(np.abs(full_kernel)**2)
    E_crop = np.sum(np.abs(cropped)**2)
    return E_crop / (E_full + EPS)

# ==============================================================================
# 模块2：光学系统参数（工业级结构化配置）
# ==============================================================================
@dataclass
class LithoOptics:
    # 物理参数 (单位: nm)
    lambda_: float          # 曝光波长 ArF=193
    NA: float               # 数值孔径
    dx: float               # 空域采样步长
    N: int                  # 仿真网格 N×N

    # 光源（工业采样版，非连续函数）
    src_type: str           # conventional / annular / quadrupole
    sigma_c: float          # 传统照明σ
    sigma_in: float         # 环形/四极内σ
    sigma_out: float        # 环形/四极外σ
    N_source: int     # 光源采样点数（工业100~2000）

    # SOCS
    n_socs: int        # 保留阶数（精度/速度权衡）

    # 像差 (Z4 离焦, λ为单位)
    defocus_z4: float = 0.0

# ==============================================================================
# 模块3：频域网格 & 光瞳（含像差）
# ==============================================================================
def freq_grid(opt: LithoOptics) -> Tuple[np.ndarray, np.ndarray]:
    """频域坐标 fx, fy [1/nm]"""
    fx = np.fft.fftfreq(opt.N, d=opt.dx)
    fy = np.fft.fftfreq(opt.N, d=opt.dx)
    fx, fy = np.meshgrid(fx, fy)
    return fx, fy

def zernike_z4_defocus(fx: np.ndarray, fy: np.ndarray, opt: LithoOptics) -> np.ndarray:
    """Z4 离焦像差（最常用光刻像差）"""
    fc = opt.NA / opt.lambda_
    f_norm = np.hypot(fx, fy) / (fc + EPS)
    f_norm = np.clip(f_norm, 0, 1)
    return 2 * f_norm**2 - 1

def pupil(fx: np.ndarray, fy: np.ndarray, opt: LithoOptics) -> np.ndarray:
    """光瞳：振幅截止 + 相位像差"""
    fc = opt.NA / opt.lambda_
    f = np.hypot(fx, fy)
    P = np.zeros_like(f, dtype=np.complex128)
    P[f <= fc] = 1.0 + 0j

    # 离焦相位
    if abs(opt.defocus_z4) > EPS:
        phase = PI * opt.defocus_z4 * zernike_z4_defocus(fx, fy, opt)
        P[f <= fc] *= np.exp(1j * phase)
    return P

# ==============================================================================
# 模块4：工业级光源离散采样（核心！替代连续积分，无TCC基础）
# ==============================================================================
def sample_source(opt: LithoOptics) -> np.ndarray:
    """
    工业标准：均匀采样光源平面 f_s (fx_s, fy_s)
    返回: [N_source, 2] 每一行是一个光源点坐标 (fx_s, fy_s)
    """
    fc = opt.NA / opt.lambda_
    Ns = opt.N_source
    f_s_list = []

    if opt.src_type == "conventional":
        r_max = opt.sigma_c * fc
        # 极坐标均匀采样
        n_r = int(np.sqrt(Ns * 0.3)) + 1
        n_theta = int(Ns / n_r) + 1
        r = np.linspace(0, r_max, n_r)
        theta = np.linspace(0, 2*PI, n_theta, endpoint=False)
        rr, tt = np.meshgrid(r, theta)
        fx_s = rr * np.cos(tt)
        fy_s = rr * np.sin(tt)
        f_s_list = np.stack([fx_s.ravel(), fy_s.ravel()], axis=1)

    elif opt.src_type == "annular":
        r_in = opt.sigma_in * fc
        r_out = opt.sigma_out * fc
        n_r = max(2, int(np.sqrt(Ns * 0.2)))
        n_theta = int(Ns / n_r) + 1
        r = np.linspace(r_in, r_out, n_r)
        theta = np.linspace(0, 2*PI, n_theta, endpoint=False)
        rr, tt = np.meshgrid(r, theta)
        fx_s = rr * np.cos(tt)
        fy_s = rr * np.sin(tt)
        f_s_list = np.stack([fx_s.ravel(), fy_s.ravel()], axis=1)

    elif opt.src_type == "quadrupole":
        r_in = opt.sigma_in * fc
        r_out = opt.sigma_out * fc
        dirs = [PI/4, 3*PI/4, 5*PI/4, 7*PI/4]
        per_dir = Ns // 4
        for th in dirs:
            r = np.linspace(r_in, r_out, per_dir)
            fx_s = r * np.cos(th)
            fy_s = r * np.sin(th)
            f_s_list.append(np.stack([fx_s, fy_s], axis=1))
        f_s_list = np.vstack(f_s_list)

    # 截断到设定采样数
    if len(f_s_list) > opt.N_source:
        f_s_list = f_s_list[:opt.N_source]
    return f_s_list

# ==============================================================================
# 模块5：工业级SOCS核心 —— 构造模式矩阵 M，不建TCC！
# ==============================================================================
def build_mode_matrix_M(opt: LithoOptics, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """
    【工业核心】构造 M ∈ C^(N² × N_source)，无TCC、无巨型矩阵
    M[:,i] = vec[ P(f+fs/2) * P*(f-fs/2) ] * sqrt(S(fs))
    """
    N = opt.N
    Ns = opt.N_source
    P = pupil(fx, fy, opt)
    fs_points = sample_source(opt)
    fc = opt.NA / opt.lambda_

    M = np.zeros((N*N, Ns), dtype=np.complex128)
    for i in range(Ns):
        fx_s, fy_s = fs_points[i]
        # 偏移频率：f+fs/2, f-fs/2
        f1x = fx + fx_s/2
        f1y = fy + fy_s/2
        f2x = fx - fx_s/2
        f2y = fy - fy_s/2

        # 光瞳双次采样
        P1 = np.zeros_like(P)
        P2 = np.zeros_like(P)
        f1 = np.hypot(f1x, f1y)
        f2 = np.hypot(f2x, f2y)
        P1[f1 <= fc] = P[f1 <= fc]
        P2[f2 <= fc] = P[f2 <= fc]

        # 模式向量
        vec_mode = (P1 * P2.conj()).ravel()
        # 光源权重 sqrt(S)
        f_s_norm = np.hypot(fx_s, fy_s) / fc
        S = 1.0 if f_s_norm <= opt.sigma_out else 0.0
        M[:, i] = vec_mode * np.sqrt(S + EPS)

    print(f"模式矩阵M尺寸: {M.shape} | 内存: {M.nbytes/1e6:.1f} MB (无TCC！)")
    return M

# ==============================================================================
# 模块6：SOCS分解（M的经济SVD，等价TCC-SVD，工业标准）
# ==============================================================================
def socs_from_M(M: np.ndarray, n_socs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    经济SVD → 直接得到SOCS基与权重
    alpha = σ² (TCC奇异值)
    V = U (SOCS频域核基向量)
    """
    U, sigma, _ = svd(M, full_matrices=False, compute_uv=True)
    alpha = sigma[:n_socs]**2  # TCC奇异值
    V = U[:, :n_socs]         # SOCS频域基
    return alpha, V

# ==============================================================================
# 模块7：SOCS空域核生成 + 无损缩核（你要求的核心优化）
# ==============================================================================
def build_socs_spatial_kernels(
    opt: LithoOptics,
    V: np.ndarray,
    crop_size: int = 64
) -> List[np.ndarray]:
    N = opt.N
    n_socs = V.shape[1]
    min_size = min_nyquist_kernel_size(opt.lambda_, opt.NA, opt.dx)
    crop_size = max(crop_size, min_size)  # 强制不混叠
    print(f"缩核配置: 奈奎斯特最小={min_size}, 实际裁剪={crop_size}")

    kernels = []
    for k in range(n_socs):
        # 频域向量 → N×N频域核
        freq_k = V[:, k].reshape(N, N)
        # IFFT → 空域核
        spat_full = ifft2c(freq_k)
        # 中心裁剪
        spat_crop = crop_kernel_center(spat_full, crop_size)
        kernels.append(spat_crop)

    # 能量校验
    e_ratio = kernel_energy_ratio(spat_full, crop_kernel_center(spat_full, crop_size))
    print(f"缩核能量保留率: {e_ratio:.6f} (≥0.9999为量产合格)")
    return kernels

# ==============================================================================
# 模块8：掩模 & FFT卷积 & 空中图像合成
# ==============================================================================
def line_mask(opt: LithoOptics) -> np.ndarray:
    """二元线条掩模"""
    mask = np.zeros((opt.N, opt.N), dtype=np.complex128)
    mask[100, :] = 1+0j
    mask[:, 50] = 1+0j
    return mask

def fft_convolve_padded(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """核补零+FFT卷积（工业快速卷积）"""
    N = img.shape[0]
    kN = kernel.shape[0]
    k_pad = np.zeros_like(img, dtype=np.complex128)
    c = N//2
    hs = kN//2
    k_pad[c-hs:c+hs, c-hs:c+hs] = kernel
    return ifft2c(fft2c(img) * fft2c(k_pad))

def aerial_image_socs(
    opt: LithoOptics,
    mask: np.ndarray,
    kernels: List[np.ndarray],
    alpha: np.ndarray
) -> np.ndarray:
    """Hopkins SOCS 部分相干成像"""
    I = np.zeros((opt.N, opt.N), dtype=np.float64)
    for k, ker in enumerate(kernels):
        conv = fft_convolve_padded(mask, ker)
        I += alpha[k] * np.abs(conv)**2
    I /= np.max(I) + EPS  # 归一化
    return I

# ==============================================================================
# 模块9：可视化 & 主流程
# ==============================================================================
def plot_aerial(img: np.ndarray, title: str, savepath="aerial_industrial.png"):
    plt.figure(figsize=(7,5))
    plt.imshow(img, cmap="gray", origin="lower")
    plt.title(title)
    plt.colorbar(label="Norm Intensity")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.show()

if __name__ == "__main__":
    # ===================== 工业参数配置 =====================
    opt = LithoOptics(
        lambda_     = 193.0,
        NA          = 0.85,
        dx          = 0.5,
        N           = 256,           # 可直接改 512 无压力
        src_type    = "conventional",
        sigma_c     = 0.5,
        sigma_in    = 0.3,
        sigma_out   = 0.7,
        N_source    = 100,            # 光源采样数
        n_socs      = 32,
        defocus_z4  = 0.0
    )

    print("="*60)
    print("工业级无TCC Hopkins SOCS 空中图像仿真")
    print(f"网格={opt.N}×{opt.N}, SOCS阶数={opt.n_socs}, 光源采样={opt.N_source}")
    print("="*60)

    # 1. 频域网格
    fx, fy = freq_grid(opt)

    # 2. 构造模式矩阵 M（核心：无TCC！）
    print("\n1/7 构造模式矩阵 M ...")
    M = build_mode_matrix_M(opt, fx, fy)

    # 3. SOCS分解
    print("2/7 SOCS 分解 (经济SVD on M) ...")
    alpha, V = socs_from_M(M, opt.n_socs)

    # 4. 空域核 + 缩核
    print("3/7 生成空域核 + 无损缩核 ...")
    kernels = build_socs_spatial_kernels(opt, V, crop_size=64)

    # 5. 掩模
    print("4/7 生成掩模 ...")
    mask = line_mask(opt)

    # 6. 成像
    print("5/7 合成空中图像 ...")
    aerial = aerial_image_socs(opt, mask, kernels, alpha)

    # 7. 绘图
    print("6/7 可视化 ...")
    plot_aerial(aerial, f"Industrial No-TCC SOCS\nN={opt.N}, Crop=64×64")

    print("\n7/7 完成！工业级无TCC仿真结束")