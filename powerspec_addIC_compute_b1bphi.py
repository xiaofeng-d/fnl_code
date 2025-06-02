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
    (1.00e11 , 3.16e11 ),
    (3.16e11 , 1.00e12 ),
    (1.00e12 , 3.16e12 ),
    (3.16e12 , 1.00e13 ),
    (1.00e13 , 3.16e13 ),
    (3.16e13 , 1.00e14 ),
    (1.00e14 , 3.16e14 ),
    (3.16e14 , 1.00e15 ),
    (1.00e15 , 1.00e16 ),
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
        k   : ndarray  (h Mpc⁻¹)
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

# def count_halos(base: Path, snap: int, mmin: float, mmax: float) -> int:
#     prefix = halos_dir(base) / f"m000-{snap}.haloproperties"
#     total = 0
#     for i in range(256):
#         fp = Path(f"{prefix}#{i}")
#         if not fp.exists():
#             continue
#         data = pygio.read_genericio(str(fp))
#         m = data["sod_halo_mass"]
#         total += int(((m > mmin) & (m < mmax)).sum())
#     if total == 0:
#         raise RuntimeError(f"No halos in {mmin:.2e}–{mmax:.2e}  (snapshot {snap})")
#     return total

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
    k, Pm, err_m = load_power(BASE_GAUSS / f"POWER/m000.pk.{SNAP}")


    # ── 新增：收集用于绘图的曲线 ─────────────────────
    curves_b1   = []      # [(label, k-array, b1-array), ...]
    points_bphi = []      # [(label, 〈b1〉, bphi), ...]
    # ────────────────────────────────────────────────


    for mmin, mmax in MASS_BINS:
        mm, mx = map(mass_str, (mmin, mmax))
        halo_pk = BASE_GAUSS / f"POWER/HALOS-b0168/m000.hh.sod.{SNAP}.ml_{mm}_mh_{mx}.pk"
        _, Ph, err_h= load_power(halo_pk)

        b1_k   = compute_b1(Ph, Pm)
        b1_bar = b1_large(k, b1_k)

        # ---------- σ8 separate‑universe halo counts ----------
        prefix_h = halos_dir(BASE_S8H) / f"m000-{SNAP}.haloproperties"
        prefix_l = halos_dir(BASE_S8L) / f"m000-{SNAP}.haloproperties"
        nh_h = len(read_halofile(prefix_h, mmin, mmax)["sod_halo_mass"])
        nh_l = len(read_halofile(prefix_l, mmin, mmax)["sod_halo_mass"])

        bphi = (np.log(nh_h) - np.log(nh_l)) / (np.log(SIGMA8_H) - np.log(SIGMA8_L))

        out = f"bias_ml{mm}_mh{mx}_snap{SNAP}.npz"
        np.savez(out, k=k, b1_k=b1_k, b1_mean=b1_bar,
                 bphi_k=np.full_like(k, bphi), bphi_mean=bphi)

        # ------- 把曲线 / 点丢进列表 -------
        label = fr"$[{mm},{mx}]$"
        curves_b1.append((label, k, b1_k))
        points_bphi.append((label, b1_bar, bphi))
        # ----------------------------------

        print(f"{out:40s}  ⟨b₁⟩={b1_bar:6.3f}   b_φ={bphi:7.3f}")

    # ============ 画图区 =============
    fig, ax1 = plt.subplots(figsize=(7,4.5))
    ax1.set_xscale('log');  ax1.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax1.set_ylabel(r"$b_1(k)$")

    for label, kk, bb in curves_b1:
        ax1.plot(kk, bb, lw=1.2, label=label)

    # 第二 y 轴画常数 b_phi
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$b_{\phi}$")
    for label, b1bar, bphi in points_bphi:
        ax2.plot([], [])                   # 只为生成图例占位
        ax2.axhline(bphi, ls='--', alpha=.4)

    # 合并图例（b1 曲线）
    ax1.legend(title=r"mass bin $[M_{\min},M_{\max}]$", fontsize=8, loc="upper left")
    plt.title(f"Snapshot {SNAP}: $b_1(k)$ and $b_\\phi$ (σ₈ SU)")
    plt.tight_layout()
    plt.savefig(f"bias_curves_snap{SNAP}.png", dpi=200)
    plt.show()
    
    #save plot
    plt.savefig(f"bias_curves_snap{SNAP}.png", dpi=200)
    # ============ 画图结束 ============

if __name__ == "__main__":
    main()