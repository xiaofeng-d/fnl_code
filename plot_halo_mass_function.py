#!/usr/bin/env python3
"""
绘制Gaussian模拟的halo mass function，特别关注1e11 M☉/h范围
Plot halo mass function for Gaussian simulation, with special focus on 1e11 M☉/h range
"""
import numpy as np
import matplotlib.pyplot as plt
import pygio
import os

def read_all_halos(haloproperties_path):
    """读取所有halo数据，不进行质量筛选
    Read all halo data without mass filtering"""
    all_data = {}
    
    # 读取第一个文件获取keys
    # Read the first file to get keys
    try:
        data_0 = pygio.read_genericio(f"{haloproperties_path}#0")
        print(f"成功读取文件 / Successfully read file: {haloproperties_path}#0")
    except Exception as e:
        print(f"无法读取文件 / Cannot read file {haloproperties_path}#0: {e}")
        return None
    
    # 初始化数据字典
    # Initialize data dictionary
    for key in data_0.keys():
        all_data[key] = []
    
    # 读取所有256个文件
    # Read all 256 files
    successful_files = 0
    for i in range(256):
        try:
            data = pygio.read_genericio(f"{haloproperties_path}#{i}")
            for key in data.keys():
                all_data[key].extend(data[key])
            successful_files += 1
        except Exception as e:
            print(f"警告: 无法读取文件 / Warning: Cannot read file #{i}: {e}")
            continue
    
    print(f"成功读取 / Successfully read {successful_files}/256 个文件 / files")
    
    # 转换为numpy数组
    # Convert to numpy arrays
    for key in all_data.keys():
        all_data[key] = np.array(all_data[key])
    
    return all_data

def calculate_hmf(halo_mass, boxsize=2000.0, n_bins=50):
    """计算halo mass function
    Calculate halo mass function"""
    # 过滤正质量的halo
    # Filter halos with positive mass
    positive_mass = halo_mass[halo_mass > 0]
    
    # 创建对数质量bins
    # Create logarithmic mass bins
    mass_bins = np.logspace(np.log10(positive_mass.min()), 
                           np.log10(positive_mass.max()), n_bins)
    
    # 计算体积
    # Calculate volume
    volume = boxsize**3  # (Mpc/h)^3
    
    # 计算histogram
    # Calculate histogram
    hist, bin_edges = np.histogram(positive_mass, bins=mass_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    dlogM = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
    
    # 转换为dn/dlogM
    # Convert to dn/dlogM
    hmf = hist / (volume * dlogM)
    
    return bin_centers, hmf

def plot_gaussian_hmf():
    """绘制Gaussian模拟的HMF
    Plot HMF for Gaussian simulation"""
    # Gaussian模拟路径
    # Gaussian simulation path
    base_path = '/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/HALOS-b0168'
    halo_file = f"{base_path}/m000-624.haloproperties"
    
    print(f"处理Gaussian模拟 / Processing Gaussian simulation")
    print(f"读取文件 / Reading file: {halo_file}")
    
    # 读取halo数据
    # Read halo data
    data = read_all_halos(halo_file)
    if data is None:
        print("无法读取数据，退出 / Cannot read data, exiting")
        return
    
    halo_mass = data["sod_halo_mass"]
    
    # 计算HMF
    # Calculate HMF
    mass_centers, hmf = calculate_hmf(halo_mass)
    
    # 创建输出目录
    # Create output directory
    output_dir = './hmf_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图形
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制HMF
    # Plot HMF
    ax.loglog(mass_centers, hmf, 'o-', label='Gaussian', 
              color='black', alpha=0.8, markersize=6, linewidth=2)
    
    # 添加重要质量标记
    # Add important mass markers
    ax.axvline(x=1e11, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label='1×10¹¹ M☉/h')
    ax.axvline(x=3.16e11, color='red', linestyle=':', alpha=0.8, linewidth=2, 
               label='3.16×10¹¹ M☉/h')
    ax.axvline(x=1e12, color='purple', linestyle='-.', alpha=0.8, linewidth=2, 
               label='1×10¹² M☉/h')
    
    # 设置图形属性
    # Set figure properties
    ax.set_xlabel('Halo Mass [M☉/h]', fontsize=14)
    ax.set_ylabel('dn/dlogM [(Mpc/h)⁻³]', fontsize=14)
    ax.set_title('Halo Mass Function - Gaussian Simulation (Step 624, z=0)', fontsize=16)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    # 设置合理的坐标轴范围
    # Set reasonable axis ranges
    ax.set_xlim(5e10, 1e15)
    ax.set_ylim(1e-8, 1e-2)
    
    # 保存图片
    # Save figures
    plt.tight_layout()
    plt.savefig(f'{output_dir}/halo_mass_function_gaussian.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/halo_mass_function_gaussian.pdf', bbox_inches='tight')
    print(f"\n图片已保存到 / Figure saved to: {output_dir}/halo_mass_function_gaussian.png")
    print(f"PDF版本已保存到 / PDF version saved to: {output_dir}/halo_mass_function_gaussian.pdf")
    
    # 关闭图形以释放内存
    # Close figure to free memory
    plt.close()
    
    # 打印统计信息
    # Print statistics
    positive_mass = halo_mass[halo_mass > 0]
    print(f"\nHalo统计信息 / Halo Statistics:")
    print(f"总halo数量 / Total halo count: {len(positive_mass):,}")
    print(f"质量范围 / Mass range: {positive_mass.min():.2e} - {positive_mass.max():.2e} M☉/h")
    print(f"质量 > 1e11 M☉/h 的halo数量 / Number of halos > 1e11 M☉/h: {np.sum(positive_mass > 1e11):,}")
    print(f"质量在 1e11-3.16e11 M☉/h 的halo数量 / Number of halos in 1e11-3.16e11 M☉/h: {np.sum((positive_mass >= 1e11) & (positive_mass <= 3.16e11)):,}")
    print(f"质量在 3.16e11-1e12 M☉/h 的halo数量 / Number of halos in 3.16e11-1e12 M☉/h: {np.sum((positive_mass >= 3.16e11) & (positive_mass <= 1e12)):,}")
    print(f"质量 > 1e12 M☉/h 的halo数量 / Number of halos > 1e12 M☉/h: {np.sum(positive_mass > 1e12):,}")
    
    # 计算不同质量范围的数密度
    # Calculate number densities for different mass ranges
    boxsize = 2000.0
    volume = boxsize**3
    
    print(f"\nHalo数密度 / Halo Number Densities:")
    print(f"总halo数密度 / Total halo number density: {len(positive_mass)/volume:.6e} (Mpc/h)⁻³")
    print(f"1e11-3.16e11 M☉/h 数密度 / 1e11-3.16e11 M☉/h number density: {np.sum((positive_mass >= 1e11) & (positive_mass <= 3.16e11))/volume:.6e} (Mpc/h)⁻³")
    print(f"3.16e11-1e12 M☉/h 数密度 / 3.16e11-1e12 M☉/h number density: {np.sum((positive_mass >= 3.16e11) & (positive_mass <= 1e12))/volume:.6e} (Mpc/h)⁻³")

def main():
    """主函数 / Main function"""
    print("绘制Gaussian模拟的Halo Mass Function / Plot Halo Mass Function for Gaussian Simulation")
    print("=" * 60)
    
    # 绘制Gaussian HMF
    # Plot Gaussian HMF
    plot_gaussian_hmf()
    
    print("\n处理完成! / Processing completed!")

if __name__ == "__main__":
    main() 