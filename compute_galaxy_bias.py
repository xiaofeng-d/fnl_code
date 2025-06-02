#!/usr/bin/env python3
"""
计算星系的线性偏差b1和增长偏差bphi，并比较物质、星系和暗晕的功率谱。
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import sys
import scipy.interpolate as interp
import matplotlib as mpl

# ────────────────── 配置 ──────────────────
SNAP = 624                      # 快照号 (624 = z0)
KMAX = 0.05                     # 线性尺度上限 (h/Mpc)

# 暗晕质量范围 (M⊙/h)
HALO_MASS_MIN = 1.00e13
HALO_MASS_MAX = 3.16e13

# 额外的暗晕质量范围 (M⊙/h)
HALO_MASS_MIN_EXTRA = 1.00e12
HALO_MASS_MAX_EXTRA = 3.16e12

# 数据目录
GALAXY_PK_DIR = "/scratch/cpac/emberson/SPHEREx/L2000/power_hod/new"
BASE_GAUSS = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/")
OUTPUT_DIR = "./results_galaxy_bias"

# 模拟参数
SIGMA8_G = 0.834   # 基准 σ₈
SIGMA8_H = 0.844   # 高 σ₈
SIGMA8_L = 0.824   # 低 σ₈

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')

# 使用系统可用的字体，避免Times New Roman字体问题
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['figure.dpi'] = 300

# ───────────────────────────────────────────────────

def mass_str(x: float) -> str:
    """将质量值转换为字符串格式"""
    return f"{x:.2e}".replace("+", "")   # 1.00e+13 → 1.00e13

def load_power(path, col_k=0, col_p=1):
    """
    读取功率谱文件，返回 k 和 P(k)
    """
    print(f"读取功率谱: {path}")
    try:
        if not os.path.exists(path):
            print(f"错误: 文件不存在 {path}")
            return None, None
            
        data = np.loadtxt(path, comments='#')
        k = data[:, col_k]
        P = data[:, col_p]
        return k, P
    except Exception as e:
        print(f"读取文件 {path} 时出错: {str(e)}")
        return None, None

def interpolate_power(k_target, k_source, P_source):
    """
    将功率谱从源k值插值到目标k值
    
    参数:
        k_target: 目标k值数组
        k_source: 源k值数组
        P_source: 源功率谱数组
    
    返回:
        P_target: 插值后的功率谱数组
    """
    # 检查k值范围
    k_min = max(k_target.min(), k_source.min())
    k_max = min(k_target.max(), k_source.max())
    
    # 创建掩码
    mask_target = (k_target >= k_min) & (k_target <= k_max)
    mask_source = (k_source >= k_min) & (k_source <= k_max)
    
    # 在对数空间中进行插值
    log_interp = interp.interp1d(
        np.log10(k_source[mask_source]), 
        np.log10(P_source[mask_source]), 
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    # 应用插值
    P_target = np.zeros_like(k_target)
    P_target[mask_target] = 10**log_interp(np.log10(k_target[mask_target]))
    
    return P_target

def compute_b1(k, Pg, Pm):
    """
    计算线性偏差 b1(k) = sqrt(Pg/Pm)
    """
    return np.sqrt(Pg / Pm)

def compute_b1_mean(k, b1_k):
    """
    计算线性区域的平均b1
    """
    mask = k <= KMAX
    return np.mean(b1_k[mask])

def compute_bphi(Pg_h, Pg_l):
    """
    计算增长偏差 bphi = d ln(Pg) / d ln(sigma8)
    """
    # 计算 ln(Pg_h/Pg_l) / ln(sigma8_h/sigma8_l)
    return np.log(Pg_h/Pg_l) / np.log(SIGMA8_H/SIGMA8_L)

def check_power_spectrum_units(k, P, name):
    """检查功率谱的基本特性"""
    print(f"\n检查 {name} 功率谱:")
    print(f"k 范围: {k.min():.6e} 到 {k.max():.6e} h/Mpc")
    print(f"P 范围: {P.min():.6e} 到 {P.max():.6e} (Mpc/h)^3")
    print(f"k 点数: {len(k)}")
    
    # 检查 P(k) 在大尺度上的行为
    large_scale_mask = k < 0.01
    if np.any(large_scale_mask):
        print(f"大尺度 P(k) (k < 0.01): {P[large_scale_mask].mean():.6e}")
    
    # 检查 P(k) 在小尺度上的行为
    small_scale_mask = k > 1.0
    if np.any(small_scale_mask):
        print(f"小尺度 P(k) (k > 1.0): {P[small_scale_mask].mean():.6e}")
    
    # 绘制功率谱
    plt.figure(figsize=(10, 6))
    plt.loglog(k, P, 'o-', lw=1, ms=3)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P(k) [(Mpc/h)^3]')
    plt.title(f'{name} Power Spectrum')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    
    # 使用安全的文件名（替换特殊字符）
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('²', '2').replace('³', '3').replace('¹', '1').replace('⊙', 'sun')
    plt.savefig(f"{OUTPUT_DIR}/{safe_name}_check.png")
    plt.close()

def plot_power_spectra(k, Pm, Pg, Ph, Ph_extra, b1_k, b1_mean, output_file):
    """
    绘制功率谱和偏差曲线
    
    参数:
        k: k值数组
        Pm: 物质功率谱
        Pg: 星系功率谱
        Ph: 暗晕功率谱 (10^13范围)
        Ph_extra: 额外的暗晕功率谱 (10^12范围)
        b1_k: 线性偏差 b1(k)
        b1_mean: 平均线性偏差
        output_file: 输出文件路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # 上图: 功率谱
    ax1.loglog(k, Pm, 'k-', lw=2, label='Matter P(k)')
    ax1.loglog(k, Pg, 'r-', lw=2, label='Galaxy P(k)')
    
    if Ph is not None:
        ax1.loglog(k, Ph, 'b--', lw=2, label='Halo P(k) (10¹³ M⊙/h)')
    
    if Ph_extra is not None:
        ax1.loglog(k, Ph_extra, 'g-.', lw=2, label='Halo P(k) (10¹² M⊙/h)')
    
    # 标记线性区域
    ax1.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # 设置坐标轴
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    ax1.set_title('Power Spectrum and Galaxy Bias', fontsize=16)
    ax1.legend(fontsize=12, frameon=True)
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    
    # 下图: 偏差
    ax2.semilogx(k, b1_k, 'r-', lw=2, label=f'b₁(k), ⟨b₁⟩ = {b1_mean:.3f}')
    
    # 标记平均偏差
    ax2.axhline(y=b1_mean, color='k', ls='--', label=f'⟨b₁⟩ = {b1_mean:.3f}')
    
    # 标记线性区域
    ax2.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # 设置y轴范围到2附近，以便更好地观察偏差
    ax2.set_ylim(0, 3)
    
    # 设置坐标轴
    ax2.set_xlabel('k [h/Mpc]', fontsize=14)
    ax2.set_ylabel('b₁(k)', fontsize=14)
    ax2.legend(fontsize=12, frameon=True)
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")
    plt.close()

def plot_all_power_spectra(k_m, Pm, Ph, Ph_extra, galaxy_results, output_file):
    """
    在一张图上绘制所有功率谱进行比较（不进行插值，显示原始曲线）
    
    参数:
        k_m: 物质功率谱的k值数组
        Pm: 物质功率谱
        Ph: 暗晕功率谱 (10^13范围)
        Ph_extra: 额外的暗晕功率谱 (不使用)
        galaxy_results: 包含各模拟星系功率谱的字典
        output_file: 输出文件路径
    """
    # 定义颜色和标记样式
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    # 绘制功率谱比较图
    plt.figure(figsize=(12, 8))
    
    # 绘制物质功率谱
    plt.loglog(k_m, Pm, 'k-', lw=2.5, label='Matter P(k)')
    
    # 绘制暗晕功率谱（只绘制10^13范围）
    if Ph is not None:
        # 重新读取原始暗晕功率谱数据
        mm, mx = map(mass_str, (HALO_MASS_MIN, HALO_MASS_MAX))
        halo_pk_file = f"{BASE_GAUSS}/POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_{mm}_mh_{mx}.pk"
        k_h_orig, Ph_orig = load_power(halo_pk_file)
        if Ph_orig is not None:
            plt.loglog(k_h_orig, Ph_orig, 'b--', lw=2.5, label='Halo P(k) (10^13 M_sun/h)')
    
    # 绘制各模拟的星系功率谱（不插值，使用原始数据）
    for i, (sim_name, pk_file) in enumerate({
        'gauss': f"{GALAXY_PK_DIR}/pk_hod_gauss.dat.pk",
        's8h': f"{GALAXY_PK_DIR}/pk_hod_s8h.dat.pk",
        's8l': f"{GALAXY_PK_DIR}/pk_hod_s8l.dat.pk",
        'fnl1': f"{GALAXY_PK_DIR}/pk_hod_fnl1.dat.pk",
        'fnl10': f"{GALAXY_PK_DIR}/pk_hod_fnl10.dat.pk"
    }.items()):
        # 重新读取原始星系功率谱数据
        k_g_orig, Pg_orig = load_power(pk_file)
        if Pg_orig is not None:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # 每隔几个点绘制一个标记以避免图形过于密集
            skip = max(1, len(k_g_orig) // 30)
            
            plt.loglog(k_g_orig[::skip], Pg_orig[::skip], 
                      color=color, marker=marker, markersize=4, 
                      linestyle='-', lw=1.5, 
                      label=f'Galaxy P(k) - {sim_name}')
    
    # 标记线性区域
    plt.axvline(x=KMAX, color='gray', ls='--', alpha=0.7, label=f'Linear limit (k={KMAX})')
    
    # 设置坐标轴
    plt.xlabel('k [h/Mpc]', fontsize=14)
    plt.ylabel('P(k) [(Mpc/h)^3]', fontsize=14)
    plt.title('Power Spectrum Comparison (Original Data)', fontsize=16)
    
    # 添加图例
    plt.legend(fontsize=10, frameon=True, loc='best')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"功率谱比较图已保存到: {output_file}")
    plt.close()

def get_galaxy_number_density():
    """
    从星系文件中直接读取星系数量，计算数密度
    """
    # 星系文件路径
    galaxy_files = {
        'gauss': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/gauss_m000-624.galaxies.haloproperties",
        's8h': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/s8h_m000-624.galaxies.haloproperties", 
        's8l': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/s8l_m000-624.galaxies.haloproperties",
        'fnl1': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/fnl1_m000-624.galaxies.haloproperties",
        'fnl10': "/home/ac.xdong/hacc-bispec/fnl-paper-plots/galaxies/fnl10_m000-624.galaxies.haloproperties"
    }
    
    # 模拟盒子体积 (Mpc/h)^3
    boxsize = 2000.0  # Mpc/h
    volume = boxsize**3
    
    # 计算数密度 (h/Mpc)^3
    galaxy_number_densities = {}
    
    for sim_name, galaxy_file in galaxy_files.items():
        try:
            import pygio
            data = pygio.read_genericio(galaxy_file)
            
            # 直接获取星系总数量（不需要质量过滤）
            # 假设每行代表一个星系，可以通过任何一个字段的长度来获取数量
            if 'x' in data:
                galaxy_count = len(data['x'])
            elif 'pos_x' in data:
                galaxy_count = len(data['pos_x'])
            else:
                # 如果不确定字段名，取第一个字段的长度
                first_key = list(data.keys())[0]
                galaxy_count = len(data[first_key])
            
            galaxy_density = galaxy_count / volume
            galaxy_number_densities[sim_name] = galaxy_density
            
            print(f"{sim_name} 星系数量: {galaxy_count}")
            print(f"{sim_name} 星系数密度: {galaxy_density:.6e} (h/Mpc)^3")
            
        except Exception as e:
            print(f"无法读取星系文件 {galaxy_file}: {e}")
            # 使用之前的估计值作为后备
            fallback_counts = {
                'gauss': 7233385,
                's8h': 7229491, 
                's8l': 7268868,
                'fnl1': 7238274,
                'fnl10': 7235730
            }
            galaxy_count = fallback_counts.get(sim_name, 7000000)  # 默认值
            galaxy_density = galaxy_count / volume
            galaxy_number_densities[sim_name] = galaxy_density
            print(f"使用后备值 - {sim_name} 星系数量: {galaxy_count}")
            print(f"使用后备值 - {sim_name} 星系数密度: {galaxy_density:.6e} (h/Mpc)^3")
    
    return galaxy_number_densities

def get_halo_number_density():
    """
    计算不同质量范围暗晕的数密度
    """
    # 读取暗晕文件获取数量
    halo_file = f"{BASE_GAUSS}/HALOS-b0168/m000-{SNAP}.haloproperties"
    
    try:
        import pygio
        data = pygio.read_genericio(halo_file)
        halo_mass = data["sod_halo_mass"]
        
        # 计算不同质量范围的暗晕数量
        mask_13 = (halo_mass >= HALO_MASS_MIN) & (halo_mass <= HALO_MASS_MAX)
        # 暂时不计算10^12范围
        # mask_12 = (halo_mass >= HALO_MASS_MIN_EXTRA) & (halo_mass <= HALO_MASS_MAX_EXTRA)
        
        count_13 = np.sum(mask_13)
        # count_12 = np.sum(mask_12)
        
        # 模拟盒子体积
        boxsize = 2000.0  # Mpc/h
        volume = boxsize**3
        
        density_13 = count_13 / volume
        # density_12 = count_12 / volume
        
        print(f"暗晕数量 (10^13范围): {count_13}")
        # print(f"暗晕数量 (10^12范围): {count_12}")
        print(f"暗晕数密度 (10^13范围): {density_13:.6e} (h/Mpc)^3")
        # print(f"暗晕数密度 (10^12范围): {density_12:.6e} (h/Mpc)^3")
        
        return {'halo_13': density_13}  # 只返回10^13范围
        
    except Exception as e:
        print(f"无法读取暗晕文件: {e}")
        # 使用估计值
        estimated_densities = {
            'halo_13': 1e-4  # 估计值，需要根据实际情况调整
        }
        return estimated_densities

def subtract_shot_noise(k, P, number_density):
    """
    从功率谱中减去shot noise
    
    参数:
        k: k值数组
        P: 功率谱数组
        number_density: 数密度 (h/Mpc)^3
    
    返回:
        P_corrected: 减去shot noise后的功率谱
    """
    shot_noise = 1.0 / number_density
    P_corrected = P - shot_noise
    
    print(f"Shot noise: {shot_noise:.6e}")
    print(f"功率谱范围 (原始): {P.min():.6e} - {P.max():.6e}")
    print(f"功率谱范围 (校正后): {P_corrected.min():.6e} - {P_corrected.max():.6e}")
    
    return P_corrected

def plot_power_spectra_shot_noise_corrected(k, Pm, Pg, Ph, Ph_extra, b1_k, b1_mean, 
                                           galaxy_density, halo_densities, sim_name, output_file):
    """
    绘制减去shot noise后的功率谱和偏差曲线
    """
    # 减去shot noise
    Pg_corrected = subtract_shot_noise(k, Pg, galaxy_density)
    
    if Ph is not None:
        Ph_corrected = subtract_shot_noise(k, Ph, halo_densities['halo_13'])
    else:
        Ph_corrected = None
        
    if Ph_extra is not None:
        Ph_extra_corrected = subtract_shot_noise(k, Ph_extra, halo_densities['halo_13'])
    else:
        Ph_extra_corrected = None
    
    # 重新计算偏差（使用校正后的星系功率谱）
    b1_k_corrected = compute_b1(k, Pg_corrected, Pm)
    b1_mean_corrected = compute_b1_mean(k, b1_k_corrected)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # 上图: 功率谱（原始和校正后）
    ax1.loglog(k, Pm, 'k-', lw=2, label='Matter P(k)')
    ax1.loglog(k, Pg, 'r--', lw=1.5, alpha=0.7, label='Galaxy P(k) (原始)')
    ax1.loglog(k, Pg_corrected, 'r-', lw=2, label='Galaxy P(k) (减去shot noise)')
    
    if Ph is not None and Ph_corrected is not None:
        ax1.loglog(k, Ph, 'b--', lw=1.5, alpha=0.7, label='Halo P(k) 10¹³ (原始)')
        ax1.loglog(k, Ph_corrected, 'b-', lw=2, label='Halo P(k) 10¹³ (减去shot noise)')
    
    if Ph_extra is not None and Ph_extra_corrected is not None:
        ax1.loglog(k, Ph_extra, 'g--', lw=1.5, alpha=0.7, label='Halo P(k) 10¹² (原始)')
        ax1.loglog(k, Ph_extra_corrected, 'g-', lw=2, label='Halo P(k) 10¹² (减去shot noise)')
    
    # 标记线性区域
    ax1.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # 设置坐标轴
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    ax1.set_title(f'Power Spectrum with Shot Noise Correction - {sim_name}', fontsize=16)
    ax1.legend(fontsize=10, frameon=True)
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    
    # 下图: 偏差比较
    ax2.semilogx(k, b1_k, 'r--', lw=1.5, alpha=0.7, label=f'b₁(k) 原始, ⟨b₁⟩ = {b1_mean:.3f}')
    ax2.semilogx(k, b1_k_corrected, 'r-', lw=2, label=f'b₁(k) 校正, ⟨b₁⟩ = {b1_mean_corrected:.3f}')
    
    # 标记平均偏差
    ax2.axhline(y=b1_mean, color='k', ls='--', alpha=0.7, label=f'⟨b₁⟩ 原始 = {b1_mean:.3f}')
    ax2.axhline(y=b1_mean_corrected, color='k', ls='-', label=f'⟨b₁⟩ 校正 = {b1_mean_corrected:.3f}')
    
    # 标记线性区域
    ax2.axvline(x=KMAX, color='gray', ls='--', alpha=0.7)
    
    # 设置y轴范围
    ax2.set_ylim(0, 3)
    
    # 设置坐标轴
    ax2.set_xlabel('k [h/Mpc]', fontsize=14)
    ax2.set_ylabel('b₁(k)', fontsize=14)
    ax2.legend(fontsize=10, frameon=True)
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Shot noise校正图表已保存到: {output_file}")
    plt.close()

def calculate_galaxy_bias():
    """计算星系偏差并比较功率谱"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取星系和暗晕数密度
    galaxy_densities = get_galaxy_number_density()
    halo_densities = get_halo_number_density()
    
    # 读取物质功率谱
    matter_pk_file = f"{BASE_GAUSS}/POWER/m000.pk.{SNAP}"
    k_m, Pm = load_power(matter_pk_file)
    if k_m is None:
        print("错误: 无法读取物质功率谱")
        return
    
    # 检查物质功率谱
    check_power_spectrum_units(k_m, Pm, "Matter")
    
    # 读取暗晕功率谱 (10^13范围)
    mm, mx = map(mass_str, (HALO_MASS_MIN, HALO_MASS_MAX))
    halo_pk_file = f"{BASE_GAUSS}/POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_{mm}_mh_{mx}.pk"
    k_h, Ph = load_power(halo_pk_file)
    
    # 检查暗晕功率谱
    if Ph is not None:
        check_power_spectrum_units(k_h, Ph, "Halo (10¹³ M⊙/h)")
        
        # 如果暗晕功率谱存在但k值与物质功率谱不同，进行插值
        if not np.array_equal(k_m, k_h):
            print("暗晕功率谱的k值与物质功率谱不同，进行插值")
            Ph = interpolate_power(k_m, k_h, Ph)
    else:
        print("警告: 无法读取暗晕功率谱")
    
    # 读取星系功率谱 - 使用新的文件路径
    galaxy_pk_files = {
        'gauss': f"{GALAXY_PK_DIR}/pk_hod_gauss.dat.pk",
        's8h': f"{GALAXY_PK_DIR}/pk_hod_s8h.dat.pk",
        's8l': f"{GALAXY_PK_DIR}/pk_hod_s8l.dat.pk",
        'fnl1': f"{GALAXY_PK_DIR}/pk_hod_fnl1.dat.pk",
        'fnl10': f"{GALAXY_PK_DIR}/pk_hod_fnl10.dat.pk"
    }
    
    # 存储结果
    results = {}
    
    # 处理每个模拟
    for sim_name, pk_file in galaxy_pk_files.items():
        print(f"\n处理模拟: {sim_name}")
        
        # 读取星系功率谱
        k_g, Pg = load_power(pk_file)
        if k_g is None:
            print(f"跳过模拟 {sim_name}: 无法读取星系功率谱")
            continue
        
        # 检查星系功率谱
        check_power_spectrum_units(k_g, Pg, f"Galaxy ({sim_name})")
        
        # 确保 k 值匹配，如果不匹配则进行插值
        if not np.array_equal(k_m, k_g):
            print(f"警告: {sim_name} 的k值与物质功率谱不同，进行插值")
            Pg = interpolate_power(k_m, k_g, Pg)
            k_g = k_m  # 使用物质功率谱的k值作为统一标准
        
        # 计算 b1(k)
        b1_k = compute_b1(k_g, Pg, Pm)
        
        # 计算线性区域的平均 b1
        b1_mean = compute_b1_mean(k_g, b1_k)
        
        # 存储结果
        results[sim_name] = {
            'k': k_g,
            'Pg': Pg,
            'b1_k': b1_k,
            'b1_mean': b1_mean
        }
        
        # 绘制原始功率谱和偏差曲线
        output_file = f"{OUTPUT_DIR}/power_bias_{sim_name}.png"
        plot_power_spectra(k_g, Pm, Pg, Ph, None, b1_k, b1_mean, output_file)
        
        # 绘制shot noise校正后的功率谱和偏差曲线
        output_file_corrected = f"{OUTPUT_DIR}/power_bias_{sim_name}_shot_noise_corrected.png"
        plot_power_spectra_shot_noise_corrected(k_g, Pm, Pg, Ph, None, b1_k, b1_mean,
                                               galaxy_densities[sim_name], halo_densities, 
                                               sim_name, output_file_corrected)
        
        print(f"{sim_name} 的平均线性偏差 b1 = {b1_mean:.3f}")
    
    # 绘制所有功率谱比较图
    all_pk_output_file = f"{OUTPUT_DIR}/all_power_spectra_comparison.png"
    plot_all_power_spectra(k_m, Pm, Ph, None, results, all_pk_output_file)

def main():
    """主函数"""
    print("计算星系偏差并比较功率谱")
    calculate_galaxy_bias()
    print("处理完成")

if __name__ == "__main__":
    main() 