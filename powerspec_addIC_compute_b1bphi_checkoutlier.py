#!/usr/bin/env python3
"""Compute ⟨b₁⟩ 和 b_φ  (σ₈ separate‑universe).

运行:  python compute_bias.py
修改:  只改 CONFIGURATION 区域即可
"""
from __future__ import annotations
import numpy as np, sys, re
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import pygio        # 读 haloproperties
except ImportError:
    sys.exit("请先 `pip install pygio`")

# ────────────────── CONFIGURATION ──────────────────
SNAP = 624                      # 快照号 (205 / 310 / 624 …)

# **手动列出想算的 mass bins** (单位 M⊙/h)
MASS_BINS = [
    (3.16e11 , 1.00e12 ),  # 这里我们关注 5.6e11 左右的质量区间
]

# MASS_BINS = [
#     (1.00e12, 3.16e12),
#     (1.00e13, 3.16e13),
#     (1.00e14, 3.16e14),
# ]

BASE_GAUSS = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0/")
BASE_S8H   = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0_s8h/")
BASE_S8L   = Path("/scratch/cpac/emberson/SPHEREx/L2000/output_l2000n4096_gauss_tpm_seed0_s8l/")

SIGMA8_H, SIGMA8_L = 0.844, 0.824   # 高 / 低 σ₈
KMAX = 0.05                         # 线性尺度上限
# ───────────────────────────────────────────────────

# ---------- 工具函数 ----------
def halos_dir(base: Path) -> Path:
    return base / "HALOS-b0168"

def mass_str(x: float) -> str:
    return f"{x:.2e}".replace("+", "")   # 1.00e+13 → 1.00e13

# ------------------------------------------------------------------
def load_power(path, col_k=0, col_p=1):
    """
    读 .pk 文件，返回
        k   : ndarray  (h Mpc⁻¹)
        P0  : ndarray  ((Mpc/h)³)
        err : ndarray  (或 None)  —— 第 3 列若存在则返回
    默认用第 1 列做 k，第 2 列做 monopole P₀(k)。
    其余列可按需改 col_k / col_p。
    """
    data = np.loadtxt(path, comments='#')        # 自动跳过任何 # 开头行
    k   = data[:, col_k]
    P0  = data[:, col_p]
    err = data[:, 2] if data.shape[1] > 2 else None
    return k, P0, err
# ------------------------------------------------------------------

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

def compute_b1(Ph, Pm):      return np.sqrt(Ph / Pm)
def b1_large(k, b1):         return float(b1[k <= KMAX].mean())

# ---------- 主流程 ----------
def main():
    # 加载物质功率谱
    k, Pm, err_m = load_power(BASE_GAUSS / f"POWER/m000.pk.{SNAP}")
    
    # 为特定质量区间绘制功率谱
    for mmin, mmax in MASS_BINS:
        mm, mx = map(mass_str, (mmin, mmax))
        halo_pk_path = BASE_GAUSS / f"POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_{mm}_mh_{mx}.pk"
        
        # 加载 halo 功率谱
        _, Ph, err_h = load_power(halo_pk_path)
        
        # 计算 b1(k)
        b1_k = compute_b1(Ph, Pm)
        b1_bar = b1_large(k, b1_k)
        
        # 创建功率谱比较图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # 上图：功率谱
        ax1.loglog(k, Pm, 'b-', lw=2, label='Matter P(k)')
        ax1.loglog(k, Ph, 'r-', lw=2, label=f'Halo P(k) [{mm}-{mx} M⊙/h]')
        ax1.axvline(KMAX, color='k', ls='--', alpha=0.5, label=f'kmax = {KMAX} h/Mpc')
        ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, which='both', ls='--', alpha=0.3)
        
        # 下图：b1(k)
        ax2.semilogx(k, b1_k, 'g-', lw=2)
        ax2.axhline(b1_bar, color='k', ls='-', alpha=0.7, 
                   label=f'⟨b₁⟩ = {b1_bar:.3f} (k ≤ {KMAX} h/Mpc)')
        ax2.axvline(KMAX, color='k', ls='--', alpha=0.5)
        ax2.set_xlabel('k [h/Mpc]', fontsize=14)
        ax2.set_ylabel('b₁(k)', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, which='both', ls='--', alpha=0.3)
        
        # 设置标题
        plt.suptitle(f'Power Spectrum and Linear Bias for Mass Range {mm}-{mx} M⊙/h', fontsize=16)
        
        # 保存图像
        output_file = f"powerspec_b1_analysis_ml{mm}_mh{mx}_snap{SNAP}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved power spectrum analysis to: {output_file}")
        
        # 保存数据以便进一步分析
        data_file = f"powerspec_data_ml{mm}_mh{mx}_snap{SNAP}.npz"
        np.savez(data_file, k=k, Pm=Pm, Ph=Ph, b1_k=b1_k, b1_mean=b1_bar, kmax=KMAX)
        print(f"Saved power spectrum data to: {data_file}")
        
        # 计算 σ8 separate-universe halo counts
        prefix_h = halos_dir(BASE_S8H) / f"m000-{SNAP}.haloproperties"
        prefix_l = halos_dir(BASE_S8L) / f"m000-{SNAP}.haloproperties"
        nh_h = len(read_halofile(prefix_h, mmin, mmax)["sod_halo_mass"])
        nh_l = len(read_halofile(prefix_l, mmin, mmax)["sod_halo_mass"])
        
        bphi = (np.log(nh_h) - np.log(nh_l)) / (np.log(SIGMA8_H) - np.log(SIGMA8_L))
        
        print(f"Mass range: {mm}-{mx} M⊙/h")
        print(f"⟨b₁⟩ = {b1_bar:.3f}")
        print(f"b_φ = {bphi:.3f}")
        print(f"Ratio b_φ/⟨b₁⟩ = {bphi/b1_bar:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    main()