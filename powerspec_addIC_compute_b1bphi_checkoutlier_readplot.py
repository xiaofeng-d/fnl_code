#!/usr/bin/env python3
"""
分析之前保存的功率谱数据，专注于 k 从 0 到 0.05 的范围。
添加 halo 数量统计和泊松噪声线。
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import sys
from pathlib import Path

try:
    import pygio  # 读 haloproperties
except ImportError:
    sys.exit("请先 `pip install pygio`")

# 配置
KMAX = 0.05  # 线性尺度上限
DATA_FILE_PATTERN = "powerspec_data_ml*.npz"  # 数据文件模式
BOX_SIZE = 2000.0  # 模拟盒子大小 (Mpc/h)
BASE_GAUSS = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/")
SNAP = 624  # 快照号

def halos_dir(base: Path) -> Path:
    """返回 HALOS 目录路径"""
    return base / "HALOS-b0168"

def format_mass(m):
    """格式化质量值，去掉科学计数法中的加号"""
    return f"{m:.2e}".replace('+', '')

def read_halofile(
        haloproperties_prefix: str | Path,
        massmin: float,
        massmax: float,
        keys_keep: tuple[str, ...] = ("sod_halo_mass",),
) -> dict[str, np.ndarray]:
    """Return dict of requested fields filtered to massmin < M < massmax."""
    prefix = Path(haloproperties_prefix)
    shards = sorted(prefix.parent.glob(prefix.name + "#*"))
    if not shards:
        raise FileNotFoundError(f"No shards match '{prefix}#*'")

    buf = {k: [] for k in keys_keep}
    for fp in shards:
        d = pygio.read_genericio(str(fp))
        m = d["sod_halo_mass"]
        sel = (m > massmin) & (m < massmax)
        if sel.any():
            for k in keys_keep:
                buf[k].append(d[k][sel])

    data = {k: np.concatenate(v) if v else np.empty(0) for k, v in buf.items()}
    return data

def get_halo_count(mmin, mmax):
    """获取指定质量范围内的 halo 数量"""
    prefix = halos_dir(BASE_GAUSS) / f"m000-{SNAP}.haloproperties"
    halos = read_halofile(prefix, mmin, mmax)
    return len(halos["sod_halo_mass"])

def calculate_poisson_noise(n_halo, box_size):
    """计算泊松噪声水平"""
    # 泊松噪声 = 1/n，其中 n 是 halo 数密度
    volume = box_size**3  # (Mpc/h)^3
    number_density = n_halo / volume  # (h/Mpc)^3
    poisson_noise = 1.0 / number_density  # (Mpc/h)^3
    return poisson_noise

def analyze_power_spectrum(data_file):
    """分析功率谱数据并绘制图表"""
    print(f"分析文件: {data_file}")
    
    # 加载数据
    data = np.load(data_file)
    k = data['k']
    Pm = data['Pm']
    Ph = data['Ph']
    b1_k = data['b1_k']
    b1_mean = data['b1_mean']
    
    # 提取质量范围信息
    match = re.search(r'ml(\d+\.\d+e[+-]?\d+)_mh(\d+\.\d+e[+-]?\d+)', data_file)
    if match:
        mmin = float(match.group(1))
        mmax = float(match.group(2))
        mm, mx = format_mass(mmin), format_mass(mmax)
        mass_range = f"{mm}-{mx} M⊙/h"
        
        # 获取 halo 数量
        n_halo = get_halo_count(mmin, mmax)
        
        # 计算泊松噪声水平
        poisson_noise = calculate_poisson_noise(n_halo, BOX_SIZE)
    else:
        mass_range = "未知质量范围"
        n_halo = 0
        poisson_noise = 0
    
    # 筛选 k <= KMAX 的数据
    mask = k <= KMAX
    k_linear = k[mask]
    Pm_linear = Pm[mask]
    Ph_linear = Ph[mask]
    b1_k_linear = b1_k[mask]
    
    # 重新计算线性区域的平均 b1
    b1_recalc = b1_k_linear.mean()
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, 
                                  gridspec_kw={'height_ratios': [2, 1]})
    
    # 上图：功率谱
    ax1.plot(k_linear, Pm_linear, 'b-', lw=2, label='Matter P(k)')
    ax1.plot(k_linear, Ph_linear, 'r-', lw=2, label=f'Halo P(k) [{mass_range}]')
    
    # 添加泊松噪声线
    if n_halo > 0:
        ax1.axhline(poisson_noise, color='gray', ls='--', alpha=0.7, 
                   label=f'Poisson noise: {poisson_noise:.2e} (Mpc/h)³')
    
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    ax1.set_yscale('log')
    
    # 下图：b1(k)
    ax2.plot(k_linear, b1_k_linear, 'g-', lw=2, label='b₁(k)')
    ax2.axhline(b1_recalc, color='k', ls='-', alpha=0.7, 
               label=f'⟨b₁⟩ = {b1_recalc:.3f} (k ≤ {KMAX} h/Mpc)')
    
    # 添加每个点的值标签
    for i, (ki, b1i) in enumerate(zip(k_linear, b1_k_linear)):
        if i % 2 == 0:  # 每隔一个点标记，避免拥挤
            ax2.annotate(f'{b1i:.3f}', (ki, b1i), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    ax2.set_xlabel('k [h/Mpc]', fontsize=14)
    ax2.set_ylabel('b₁(k)', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    # 设置 x 轴范围为 0 到 KMAX
    ax1.set_xlim(0, KMAX)
    
    # 设置标题
    halo_info = f"Halos: {n_halo:,}" if n_halo > 0 else ""
    plt.suptitle(f'Power Spectrum and Linear Bias (k ≤ {KMAX} h/Mpc)\n{mass_range} {halo_info}', fontsize=16)
    
    # 保存图像
    output_file = f"linear_range_analysis_{os.path.basename(data_file).replace('.npz', '.png')}"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"保存分析图表到: {output_file}")
    
    # 打印详细信息
    print("\n质量范围信息:")
    print(f"质量范围: {mass_range}")
    print(f"Halo 数量: {n_halo:,}")
    print(f"泊松噪声水平: {poisson_noise:.2e} (Mpc/h)³")
    
    print("\nb1(k) 值 (k ≤ KMAX):")
    print(f"{'k [h/Mpc]':12s} {'b1(k)':10s}")
    print("-" * 25)
    for ki, b1i in zip(k_linear, b1_k_linear):
        print(f"{ki:12.6f} {b1i:10.3f}")
    
    print(f"\n平均 b1 = {b1_recalc:.3f}")
    print("-" * 50)

def main():
    """主函数"""
    # 查找所有匹配的数据文件
    data_files = sorted(glob.glob(DATA_FILE_PATTERN))
    
    if not data_files:
        print(f"未找到匹配 '{DATA_FILE_PATTERN}' 的数据文件")
        return
    
    print(f"找到 {len(data_files)} 个数据文件")
    
    # 分析每个数据文件
    for data_file in data_files:
        analyze_power_spectrum(data_file)

if __name__ == "__main__":
    main()