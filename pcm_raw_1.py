import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.interpolate import griddata, Rbf
from dataclasses import dataclass
from typing import Tuple, List
import time
import gc

# ==============================================================================
# Module 1: Core Utilities for Rectangular Grids
# ==============================================================================
PI = np.pi
EPS = 1e-12

def fft2c_rect(x: np.ndarray) -> np.ndarray:
    """Centered FFT2 for rectangular grids"""
    return np.fft.fftshift(np.fft.fft2(x))

def ifft2c_rect(x: np.ndarray) -> np.ndarray:
    """Centered IFFT2 for rectangular grids"""
    return np.fft.ifft2(np.fft.ifftshift(x))

def crop_kernel_center_rect(kernel: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Crop kernel to centered rectangular region (supports asymmetric dimensions)"""
    H, W = kernel.shape
    c_h, c_w = H // 2, W // 2
    hs_h, hs_w = crop_h // 2, crop_w // 2
    return kernel[c_h-hs_h:c_h+hs_h, c_w-hs_w:c_w+hs_w]

def pad_kernel_to_size_rect(kernel: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad kernel to target rectangular size with zero-padding"""
    H, W = kernel.shape
    padded = np.zeros((target_h, target_w), dtype=kernel.dtype)
    c_h, c_w = target_h // 2, target_w // 2
    hs_h, hs_w = H // 2, W // 2
    padded[c_h-hs_h:c_h+hs_h, c_w-hs_w:c_w+hs_w] = kernel
    return padded

def min_nyquist_kernel_size_rect(lambda_nm: float, NA: float, dx_nm: float, dy_nm: float) -> Tuple[int, int]:
    """Calculate minimum odd-sized kernel dimensions based on Nyquist criterion"""
    fc = NA / lambda_nm
    min_px_h = np.ceil(1.0 / (dy_nm * 2 * fc))
    min_px_w = np.ceil(1.0 / (dx_nm * 2 * fc))
    min_h = int(min_px_h) if min_px_h % 2 == 1 else int(min_px_h + 1)
    min_w = int(min_px_w) if min_px_w % 2 == 1 else int(min_px_w + 1)
    return min_h, min_w

# ==============================================================================
# Module 2: Lithography Optical Parameters (355nm Configuration)
# ==============================================================================
@dataclass
class LithoOpticsRect:
    # Physical Parameters (User Configurable)
    lambda_nm: float = 355.0          # Source wavelength [nm]
    NA: float = 0.686                 # Numerical aperture
    dx_nm: float = 150.0              # X-axis spatial sampling [nm]
    dy_nm: float = 150.0              # Y-axis spatial sampling [nm]
    N_x: int = 2048                   # X-axis grid points
    N_y: int = 1024                   # Y-axis grid points (asymmetric)
    sigma_c: float = 0.5              # Conventional source coherence factor
    
    # Mask Pattern Parameters
    line_width_nm: float = 3000.0     # Line width for test pattern
    pitch_nm: float = 12000.0         # Pitch period for test pattern
    
    # Algorithmic Parameters
    N_source: int = 200               # Number of source sampling points
    n_socs: int = 32                  # Number of SOCS retained orders
    kernel_crop_h: int = 64           # Kernel crop height (Y-axis)
    kernel_crop_w: int = 64           # Kernel crop width (X-axis)
    
    # Sparse Sampling Parameter
    sparse_sampling_rate: float = 0.30  # Sampling ratio within optical support (0-1)
    
    # Interpolation Method
    interpolation_method: str = 'rbf'   # 'rbf', 'linear', 'nearest'
    
    # Reproducibility
    random_seed: int = 42               # Random seed for deterministic sampling
    
    def __post_init__(self):
        assert self.N_x % 2 == 0 and self.N_y % 2 == 0, "Grid dimensions must be even"
        print(f"[CONFIG] Rectangular Grid: {self.N_y}×{self.N_x} | λ={self.lambda_nm}nm NA={self.NA:.3f} dx={self.dx_nm}nm")
        
        # Calculate optical support ratio
        fc = self.NA / self.lambda_nm
        f_max = 1/(2*self.dx_nm)
        support_ratio = PI * (fc * (1+self.sigma_c))**2 / (2*f_max)**2
        print(f"[OPTICS] Cutoff Frequency: {fc:.5f} 1/nm | Support Ratio: {support_ratio*100:.2f}%")
        
        # Enforce Nyquist constraint on kernel size
        min_h, min_w = min_nyquist_kernel_size_rect(self.lambda_nm, self.NA, self.dx_nm, self.dy_nm)
        if self.kernel_crop_h < min_h or self.kernel_crop_w < min_w:
            print(f"[WARNING] Kernel size too small. Auto-adjusting: {self.kernel_crop_h}×{self.kernel_crop_w} → {max(min_h, self.kernel_crop_h)}×{max(min_w, self.kernel_crop_w)}")
            self.kernel_crop_h = max(min_h, self.kernel_crop_h)
            self.kernel_crop_w = max(min_w, self.kernel_crop_w)

# ==============================================================================
# Module 3: Frequency Grid and Optical Support
# ==============================================================================
def freq_grid_rect(opt: LithoOpticsRect) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rectangular frequency coordinate grid [1/nm]"""
    fx = np.fft.fftfreq(opt.N_x, d=opt.dx_nm)
    fy = np.fft.fftfreq(opt.N_y, d=opt.dy_nm)
    return np.meshgrid(fx, fy)

def get_optical_support(opt: LithoOpticsRect, fx: np.ndarray, fy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify valid frequency indices within optical support
    Returns: valid_indices, support_mask
    """
    fc = opt.NA / opt.lambda_nm
    f_bound = fc * (1 + opt.sigma_c)  # Consider partial coherence extension
    f_rho = np.hypot(fx, fy)
    support_mask = f_rho <= f_bound
    
    valid_indices = np.where(support_mask.ravel())[0]
    print(f"[COMPRESS-1] Optical Support: {opt.N_y*opt.N_x} → {len(valid_indices)} points ({len(valid_indices)/(opt.N_y*opt.N_x)*100:.2f}%)")
    return valid_indices, support_mask

# ==============================================================================
# Module 4: Adaptive Sparse Frequency Sampling (CRITICAL BUG FIX)
# ==============================================================================
def adaptive_sparse_freq_sampling(opt: LithoOpticsRect, fx: np.ndarray, fy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive sparse sampling: non-uniform points within optical support
    
    BUG FIX: Correctly handles fftfreq ordering and ensures no shape mismatches.
    Returns flattened 1D arrays for consistent matrix operations.
    """
    fc = opt.NA / opt.lambda_nm
    f_bound = fc * (1 + opt.sigma_c)
    
    # Get 1D frequency arrays (unique values)
    fx_1d = np.fft.fftfreq(opt.N_x, d=opt.dx_nm)
    fy_1d = np.fft.fftfreq(opt.N_y, d=opt.dy_nm)
    
    # Find VALID indices where |frequency| <= bound
    # This is the correct way to identify optical support indices
    fx_valid_mask = np.abs(fx_1d) <= f_bound
    fy_valid_mask = np.abs(fy_1d) <= f_bound
    
    fx_valid_indices = np.where(fx_valid_mask)[0]
    fy_valid_indices = np.where(fy_valid_mask)[0]
    
    # Adaptive sampling: denser near center (zero frequency)
    # Use non-linear sampling (cubic root) to concentrate points near center
    n_fx_sample = max(int(len(fx_valid_indices) * opt.sparse_sampling_rate), 10)
    n_fy_sample = max(int(len(fy_valid_indices) * opt.sparse_sampling_rate), 10)
    
    # Sample with higher density near zero frequency
    # Use cumulative distribution of a Gaussian-like function
    fx_center_idx = len(fx_valid_indices) // 2
    fy_center_idx = len(fy_valid_indices) // 2
    
    # Generate sample indices using non-linear spacing
    indices = np.linspace(0, 1, n_fx_sample)**0.3  # Concentrate near start
    fx_sample_indices = (indices * (len(fx_valid_indices)-1)).astype(int)
    fx_sample_indices = np.unique(fx_sample_indices)
    
    indices = np.linspace(0, 1, n_fy_sample)**0.3
    fy_sample_indices = (indices * (len(fy_valid_indices)-1)).astype(int)
    fy_sample_indices = np.unique(fy_sample_indices)
    
    # Map back to original array indices
    fx_indices_to_use = fx_valid_indices[fx_sample_indices]
    fy_indices_to_use = fy_valid_indices[fy_sample_indices]
    
    # Create sparse grids (2D for interpolation compatibility)
    fx_sparse, fy_sparse = np.meshgrid(fx_1d[fx_indices_to_use], fy_1d[fy_indices_to_use])
    
    # Calculate sampling weights to compensate for undersampling
    sample_ratio_x = len(fx_indices_to_use) / opt.N_x
    sample_ratio_y = len(fy_indices_to_use) / opt.N_y
    weights_factor = 1.0 / (sample_ratio_x * sample_ratio_y)
    sampling_weights = np.full_like(fx_sparse, weights_factor).ravel()
    
    print(f"[COMPRESS-2] Sparse Sampling: {opt.N_y}×{opt.N_x} → {len(fy_indices_to_use)}×{len(fx_indices_to_use)} = {len(fx_indices_to_use)*len(fy_indices_to_use)} points")
    print(f"             Sampling Rate: {opt.sparse_sampling_rate*100:.0f}% | Compression: {opt.N_y*opt.N_x/(len(fx_indices_to_use)*len(fy_indices_to_use)):.1f}x")
    
    # Return flattened arrays to prevent shape mismatches in matrix operations
    return fx_sparse.ravel(), fy_sparse.ravel(), sampling_weights

# ==============================================================================
# Module 5: Pupil and Source Functions
# ==============================================================================
def pupil_rect(fx: np.ndarray, fy: np.ndarray, opt: LithoOpticsRect) -> np.ndarray:
    """Rectangular pupil function (supports sparse/full grid)"""
    fc = opt.NA / opt.lambda_nm
    f_rho = np.hypot(fx, fy)
    P = np.zeros_like(f_rho, dtype=np.complex128)
    valid = f_rho <= fc
    P[valid] = 1.0 + 0j
    return P

def shifted_pupil_rect(fx: np.ndarray, fy: np.ndarray, f_shift: np.ndarray, opt: LithoOpticsRect) -> np.ndarray:
    """Shifted pupil function for oblique incidence"""
    fc = opt.NA / opt.lambda_nm
    fx_shifted = fx - f_shift[0]
    fy_shifted = fy - f_shift[1]
    f_rho = np.hypot(fx_shifted, fy_shifted)
    P = np.zeros_like(f_rho, dtype=np.complex128)
    valid = f_rho <= fc
    P[valid] = 1.0 + 0j
    return P

def sample_conventional_source(opt: LithoOpticsRect) -> Tuple[np.ndarray, np.ndarray]:
    """Sample conventional source (annular/circular) uniformly in area"""
    fc = opt.NA / opt.lambda_nm
    max_radius = opt.sigma_c * fc
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(opt.random_seed)
    
    # Uniform sampling in circle using area-preserving transformation
    u = rng.rand(opt.N_source)
    r = np.sqrt(u) * max_radius
    theta = rng.rand(opt.N_source) * 2 * PI
    
    fx_s = r * np.cos(theta)
    fy_s = r * np.sin(theta)
    
    # Equal weights for uniform sampling
    weights = np.ones(opt.N_source) / opt.N_source
    return np.stack([fx_s, fy_s], axis=1), weights

# ==============================================================================
# Module 6: Sparse Mode Matrix Construction (CRITICAL BUG FIX)
# ==============================================================================
def build_sparse_mode_matrix_M(
    opt: LithoOpticsRect,
    fx_sparse: np.ndarray,
    fy_sparse: np.ndarray,
    sampling_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sparse M matrix: M_sparse ∈ C^(N_sparse × N_source)
    
    CRITICAL FIX: All inputs are now 1D flattened arrays, preventing shape mismatches.
    """
    N_sparse = len(fx_sparse)  # Total number of sparse frequency points
    Ns = opt.N_source
    
    # Sample source points
    fs_coords, weights = sample_conventional_source(opt)
    
    # Pre-allocate matrix
    M_sparse = np.zeros((N_sparse, Ns), dtype=np.complex128)
    
    # Pre-calculate sqrt of sampling_weights for efficiency
    sqrt_sampling_weights = np.sqrt(sampling_weights + EPS)
    
    print(f"[BUILD] Constructing M matrix...")
    start = time.time()
    
    for i in range(Ns):
        fx_s, fy_s = fs_coords[i]
        # Combine source weight with sampling weights (element-wise)
        # sampling_weights is 1D, weights[i] is scalar
        w = sqrt_sampling_weights * np.sqrt(weights[i] + EPS)
        
        # Calculate shifted pupils on flattened sparse grid
        P1 = shifted_pupil_rect(fx_sparse, fy_sparse, np.array([-fx_s/2, -fy_s/2]), opt)
        P2 = shifted_pupil_rect(fx_sparse, fy_sparse, np.array([+fx_s/2, +fy_s/2]), opt)
        
        # Calculate sparse mode vector (all 1D arrays, element-wise multiplication)
        mode_sparse = P1 * P2.conj() * w
        
        # Assign to column
        M_sparse[:, i] = mode_sparse
        
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{Ns} modes completed")
    
    elapsed = time.time() - start
    print(f"[BUILD] M matrix shape: {M_sparse.shape}, Memory: {M_sparse.nbytes/1e6:.1f} MB | Time: {elapsed:.2f}s")
    
    return M_sparse, fs_coords, weights

# ==============================================================================
# Module 7: Sparse Domain SVD Decomposition
# ==============================================================================
def sparse_socs_decompose(M_sparse: np.ndarray, n_socs: int, energy_threshold: float = 0.9999) -> Tuple[np.ndarray, np.ndarray]:
    """
    Economic SVD decomposition in sparse domain
    """
    print(f"\n[SVD] Sparse domain decomposition...")
    start = time.time()
    
    # Use LAPACK's gesvd for better numerical stability
    U, sigma, _ = svd(M_sparse, full_matrices=False, overwrite_a=True, check_finite=False)
    alpha = sigma[:n_socs]**2
    
    # Energy validation
    total_energy = np.sum(sigma**2)
    retained_energy = np.sum(alpha)
    energy_ratio = retained_energy / (total_energy + EPS)
    
    print(f"      Energy Retained: {energy_ratio:.6f} (Target ≥ {energy_threshold})")
    if energy_ratio < energy_threshold:
        recommended_n_socs = int(np.ceil(n_socs / energy_ratio))
        print(f"      WARNING: Consider increasing n_socs to {recommended_n_socs}")
    
    V_sparse = U[:, :n_socs]  # Sparse basis
    print(f"      SVD Time: {time.time()-start:.2f}s")
    
    return alpha, V_sparse

# ==============================================================================
# Module 8: Kernel Reconstruction from Sparse Basis (CRITICAL BUG FIX)
# ==============================================================================
def reconstruct_kernels_from_sparse(
    opt: LithoOpticsRect,
    V_sparse: np.ndarray,
    fx_sparse: np.ndarray,
    fy_sparse: np.ndarray,
    sampling_weights: np.ndarray,
) -> List[np.ndarray]:
    """
    Reconstruct full-size spatial kernels from sparse basis using interpolation
    
    BUG FIX: Added fill_value=0.0 to griddata to prevent NaN propagation.
    Added explicit NaN check and handling for robustness.
    """
    Ny, Nx = opt.N_y, opt.N_x
    n_socs = V_sparse.shape[1]
    
    # Full frequency grid
    fx_full, fy_full = freq_grid_rect(opt)
    
    kernels = []
    print(f"\n[RECON] Interpolating {n_socs} kernels...")
    print(f"         Method: {opt.interpolation_method}")
    start = time.time()
    
    # Prepare sparse points for interpolation (2D coordinates)
    points_sparse = np.column_stack([fx_sparse, fy_sparse])
    
    for k in range(n_socs):
        Vk_values = V_sparse[:, k]
        
        # Interpolate to full grid
        if opt.interpolation_method == 'rbf':
            # Radial Basis Function (accurate but slower)
            # Use smaller epsilon for smoother interpolation
            rbf_interp = Rbf(
                fx_sparse, 
                fy_sparse, 
                Vk_values,
                function='multiquadric',
                epsilon=0.15
            )
            Vk_full = rbf_interp(fx_full, fy_full)
        else:
            # griddata interpolation (faster)
            # CRITICAL FIX: Added fill_value=0.0 to prevent NaN outside convex hull
            Vk_full = griddata(
                points_sparse, 
                Vk_values, 
                (fx_full, fy_full), 
                method=opt.interpolation_method,
                fill_value=0.0  # Fill outside support with 0 (no high-frequency content)
            )
        
        # CRITICAL FIX: Explicit NaN check and handling
        if np.any(np.isnan(Vk_full)):
            print(f"[WARNING] NaN detected in interpolated Vk_full for kernel {k+1}. Filling with 0.")
            Vk_full = np.nan_to_num(Vk_full, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Transform to spatial domain
        spat_full = ifft2c_rect(Vk_full) * np.sqrt(Ny*Nx)
        
        # Center crop
        spat_crop = crop_kernel_center_rect(spat_full, opt.kernel_crop_h, opt.kernel_crop_w)
        
        kernels.append(spat_crop)
        
        if (k+1) % 10 == 0:
            print(f"  Reconstructed {k+1}/{n_socs} kernels")
    
    print(f"[RECON] Reconstruction Time: {time.time()-start:.2f}s")
    return kernels

# ==============================================================================
# Module 9: Mask and Imaging
# ==============================================================================
def binary_line_mask_rect(opt: LithoOpticsRect) -> np.ndarray:
    """Generate binary line/space mask pattern (X-oriented lines)"""
    x = (np.arange(opt.N_x) - opt.N_x//2) * opt.dx_nm
    y = (np.arange(opt.N_y) - opt.N_y//2) * opt.dy_nm
    xx, yy = np.meshgrid(x, y)
    
    # X-oriented periodic lines
    mod = np.mod(xx + opt.pitch_nm/2, opt.pitch_nm)
    mask = (mod < opt.line_width_nm).astype(np.complex128)
    return mask

def fft_convolve_rect(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT-based convolution for rectangular grids"""
    Ny, Nx = img.shape
    kernel_padded = pad_kernel_to_size_rect(kernel, Ny, Nx)
    
    # Apply FFT and multiply in frequency domain
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
    """Generate aerial image using Hopkins SOCS method"""
    I = np.zeros((opt.N_y, opt.N_x), dtype=np.float64)
    
    print(f"\n[IMAGING] SOCS synthesis ({opt.n_socs} orders)...")
    start = time.time()
    
    # Accumulate intensity contributions from each kernel
    for k, ker in enumerate(kernels):
        conv = fft_convolve_rect(mask, ker)
        I += alpha[k] * np.abs(conv)**2
        
        if (k+1) % 10 == 0:
            print(f"  Order {k+1}/{len(kernels)} completed")
    
    # Normalize intensity
    I_norm = I / (np.max(I) + EPS)
    print(f"[IMAGING] Imaging Time: {time.time()-start:.2f}s")
    
    return I_norm

# ==============================================================================
# Module 10: Complete Compressed Simulation Pipeline
# ==============================================================================
def run_ultra_compressed_simulation(opt: LithoOpticsRect) -> dict:
    """
    Ultra-compressed simulation pipeline:
    1. Optical support compression
    2. Frequency domain sparse sampling
    3. Interpolated kernel reconstruction
    """
    print("\n" + "="*70)
    print("LAUNCHING ULTRA-COMPRESSED INDUSTRIAL HOPKINS SOCS LITHOGRAPHY SIMULATION")
    print("="*70)
    start_total = time.time()
    
    # 1. Full frequency grid (for interpolation reference)
    print("\n[STEP 1/6] Generating full-size frequency grid...")
    fx_full, fy_full = freq_grid_rect(opt)
    
    # 2. Optical support compression
    print("\n[STEP 2/6] Identifying optical support region...")
    valid_indices, support_mask = get_optical_support(opt, fx_full, fy_full)
    
    # 3. Adaptive sparse sampling
    print("\n[STEP 3/6] Performing adaptive sparse sampling...")
    fx_sparse, fy_sparse, sampling_weights = adaptive_sparse_freq_sampling(opt, fx_full, fy_full)
    
    # 4. Build sparse M matrix (critical memory optimization)
    print("\n[STEP 4/6] Building sparse M matrix...")
    M_sparse, fs_coords, weights = build_sparse_mode_matrix_M(opt, fx_sparse, fy_sparse, sampling_weights)
    
    # 5. Sparse domain SVD
    print("\n[STEP 5/6] Executing sparse domain SVD...")
    alpha, V_sparse = sparse_socs_decompose(M_sparse, opt.n_socs)
    
    # 6. Reconstruct spatial kernels
    print("\n[STEP 6/6] Reconstructing kernels via interpolation...")
    kernels = reconstruct_kernels_from_sparse(opt, V_sparse, fx_sparse, fy_sparse, sampling_weights)
    
    # 7. Mask and imaging
    print("\n[PROCESSING] Generating mask pattern...")
    mask = binary_line_mask_rect(opt)
    aerial = aerial_image_socs_rect(opt, mask, kernels, alpha)
    
    # 8. Visualization
    visualize_compressed_results(opt, mask, aerial, kernels, alpha, M_sparse)
    
    print(f"\n✅ SIMULATION COMPLETE! Total Time: {time.time()-start_total:.2f}s")
    
    return {
        "opt": opt,
        "mask": mask,
        "aerial": aerial,
        "kernels": kernels,
        "alpha": alpha,
        "M_sparse": M_sparse,
        "fx_sparse": fx_sparse,
        "fy_sparse": fy_sparse
    }

# ==============================================================================
# Module 11: Visualization and Quality Report
# ==============================================================================
def visualize_compressed_results(
    opt: LithoOpticsRect,
    mask: np.ndarray,
    aerial: np.ndarray,
    kernels: List[np.ndarray],
    alpha: np.ndarray,
    M_sparse: np.ndarray,
    save_path: str = "litho_compressed_355nm.png"
):
    """Generate comprehensive visualization and quality report"""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Mask Pattern
    ax1 = plt.subplot(3, 3, 1)
    plt.imshow(np.abs(mask), cmap="gray", origin="lower", aspect='auto')
    plt.title(f"Mask Pattern: {opt.N_y}×{opt.N_x} pixels\nLine Width: {opt.line_width_nm}nm, Pitch: {opt.pitch_nm}nm")
    plt.colorbar(label="Transmission")
    
    # 2. Aerial Image
    ax2 = plt.subplot(3, 3, 2)
    im = plt.imshow(aerial, cmap="gray", origin="lower", aspect='auto', vmin=0, vmax=1)
    plt.title("Aerial Image (SOCS Synthesis)")
    plt.colorbar(im, label="Normalized Intensity")
    
    # 3. X-direction Cross-Section (along pitch)
    ax3 = plt.subplot(3, 3, 3)
    center_y = opt.N_y // 2
    profile_x = aerial[center_y, :]
    plt.plot(profile_x, 'b-', linewidth=2)
    contrast = profile_x.max() - profile_x.min() if profile_x.max() > 0 else 0
    plt.title(f"Cross-Section at Y={center_y}\nContrast: {contrast:.3f}")
    plt.xlabel("X Pixel Index")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.3)
    
    # 4. Y-direction Cross-Section (across lines)
    ax4 = plt.subplot(3, 3, 4)
    center_x = opt.N_x // 4  # Avoid periodic nulls
    profile_y = aerial[:, center_x]
    plt.plot(profile_y, 'r-', linewidth=2)
    plt.title(f"Cross-Section at X={center_x}")
    plt.xlabel("Y Pixel Index")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.3)
    
    # 5-7. First 3 SOCS Kernels
    for i in range(min(3, len(kernels))):
        ax = plt.subplot(3, 3, 5+i)
        ker = np.abs(kernels[i])
        plt.imshow(ker, cmap="hot", origin="lower")
        plt.title(f"Kernel #{i+1}\nα={alpha[i]:.2e}")
        plt.colorbar(label="Magnitude")
    
    # 8. Energy Distribution
    ax8 = plt.subplot(3, 3, 8)
    plt.semilogy(alpha / alpha.sum(), 'ko-', markersize=8)
    plt.title("SOCS Energy Distribution")
    plt.xlabel("Kernel Order")
    plt.ylabel("Relative Energy")
    plt.grid(True, which="both", alpha=0.3)
    
    # 9. Memory Optimization
    ax9 = plt.subplot(3, 3, 9)
    memory_full = (opt.N_y * opt.N_x * opt.N_source * 16) / 1e9
    memory_sparse = M_sparse.nbytes / 1e9
    bars = plt.bar(['Full Matrix', 'Compressed'], [memory_full, memory_sparse], color=['red', 'green'])
    plt.title("Memory Usage Comparison")
    plt.ylabel("Memory [GB]")
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}GB\n({memory_full/memory_sparse:.0f}x)', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    # Quality Report
    print("\n" + "="*70)
    print("INDUSTRIAL ULTRA-COMPRESSED LITHOGRAPHY SIMULATION QUALITY REPORT")
    print("="*70)
    print(f"Optical Parameters: λ={opt.lambda_nm}nm, NA={opt.NA:.3f}, σ={opt.sigma_c:.2f}")
    print(f"Grid Configuration: {opt.N_y}×{opt.N_x} pixels | dx={opt.dx_nm}nm, dy={opt.dy_nm}nm")
    print(f"Mask Pattern: Line Width={opt.line_width_nm}nm, Pitch={opt.pitch_nm}nm")
    print(f"SOCS Configuration: Orders={opt.n_socs} | Source Samples={opt.N_source}")
    print(f"Kernel Size: {opt.kernel_crop_h}×{opt.kernel_crop_w} pixels")
    print(f"Interpolation Method: {opt.interpolation_method}")
    print(f"\nCompression Performance:")
    print(f"  M-Matrix Memory: {M_sparse.nbytes/1e6:.1f} MB (Theoretical Full: {(opt.N_y*opt.N_x*opt.N_source*16)/1e6:.1f} MB)")
    print(f"  Compression Ratio: {(opt.N_y*opt.N_x*opt.N_source)/M_sparse.shape[0]:.0f}x")
    print(f"\nImage Quality:")
    contrast = aerial.max() - aerial.min() if aerial.max() > 0 else 0
    print(f"  Image Contrast: {contrast:.4f}")
    print(f"  Dynamic Range: [{aerial.min():.4f}, {aerial.max():.4f}]")
    print("="*70)

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # ============== 355nm/150nm/0.686 Configuration ==============
    opt_355nm = LithoOpticsRect(
        lambda_nm=355.0,
        NA=0.686,
        dx_nm=150.0,
        dy_nm=150.0,
        N_x=2048,
        N_y=1024,
        sigma_c=0.5,
        N_source=200,
        n_socs=32,
        kernel_crop_h=64,
        kernel_crop_w=64,
        sparse_sampling_rate=0.30,  # 30% sampling rate (tuneable: 0.20-0.50)
        interpolation_method='rbf',  # 'rbf', 'linear', 'nearest'
        random_seed=42
    )
    
    # Configure test pattern
    opt_355nm.line_width_nm = 3000.0   # 3μm line width
    opt_355nm.pitch_nm = 12000.0       # 12μm pitch
    
    # Run simulation
    results = run_ultra_compressed_simulation(opt_355nm)
    
    # Clean up large arrays
    gc.collect()
    print("[CLEANUP] Large arrays cleared from memory.")
