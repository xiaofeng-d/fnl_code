#!/usr/bin/env python3
"""
重新绘制 HOD 对比图，使用之前保存的数据。
这个脚本加载 .npz 文件并创建高质量的可视化图表。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
import argparse
from pathlib import Path

# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['figure.dpi'] = 300

def find_latest_npz(pattern="*_arrays.npz"):
    """查找最新的 npz 文件"""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"找不到匹配 '{pattern}' 的文件")
    
    # 按修改时间排序，返回最新的
    return max(files, key=os.path.getmtime)

def plot_hod_comparison(data_file, output_file=None, title=None):
    """绘制 HOD 对比图"""
    print(f"加载数据文件: {data_file}")
    data = np.load(data_file)
    
    # 提取数据
    bin_centers = data['bin_centers']
    avg_ncen = data['avg_ncen']
    avg_nsat = data['avg_nsat']
    err_ncen = data['err_ncen']
    err_nsat = data['err_nsat']
    M_grid = data['M_grid']
    N_cen_analytic = data['N_cen_analytic']
    N_sat_analytic = data['N_sat_analytic']
    
    # 打印一些统计信息
    print(f"质量范围: {bin_centers.min():.2e} 到 {bin_centers.max():.2e}")
    print(f"平均中心星系数量范围: {min(avg_ncen):.4f} 到 {max(avg_ncen):.4f}")
    print(f"平均卫星星系数量范围: {min(avg_nsat):.4f} 到 {max(avg_nsat):.4f}")
    
    # 创建图表
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # 绘制解析曲线
    ax.loglog(M_grid, N_cen_analytic, 'k-', lw=2.5, 
              label="Rocher+25 analytical model $N_{\\mathrm{cen}}$")
    ax.loglog(M_grid, N_sat_analytic, 'k--', lw=2.5, 
              label="Rocher+25 analytical model $N_{\\mathrm{sat}}$")
    
    # 绘制模拟数据点
    ax.errorbar(bin_centers, avg_ncen, yerr=err_ncen, 
                fmt='o', ms=8, color='#1f77b4', capsize=4, elinewidth=2,
                label="sims $\\langle N_{\\mathrm{cen}}\\rangle$")
    ax.errorbar(bin_centers, avg_nsat, yerr=err_nsat, 
                fmt='s', ms=8, color='#ff7f0e', capsize=4, elinewidth=2,
                label="sims $\\langle N_{\\mathrm{sat}}\\rangle$")
    
    # 设置坐标轴
    ax.set_xlabel(r"$M_{200c}\;[h^{-1}M_\odot]$", fontsize=16)
    ax.set_ylabel(r"$\langle N\,|\,M\rangle$", fontsize=16)
    ax.set_xlim(1e11, 1e15)
    ax.set_ylim(1e-7, 1e2)
    
    # 添加网格和图例
    # ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=14, loc='upper left')
    
    # 设置标题
    if title:
        ax.set_title(title, fontsize=18)
    else:
        ax.set_title("galaxy HOD model comparison: Rocher+25", fontsize=18)
    
    # 保存图表
    if output_file is None:
        output_file = Path(data_file).stem.replace('_arrays', '') + '_enhanced2.png'
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="重新绘制 HOD 对比图")
    parser.add_argument('--file', '-f', help='数据文件路径 (.npz)')
    parser.add_argument('--output', '-o', help='输出图像文件路径')
    parser.add_argument('--title', '-t', help='图表标题')
    args = parser.parse_args()
    
    # 如果未指定文件，查找最新的
    data_file = args.file if args.file else find_latest_npz()
    
    plot_hod_comparison(data_file, args.output, args.title)

if __name__ == "__main__":
    main() 