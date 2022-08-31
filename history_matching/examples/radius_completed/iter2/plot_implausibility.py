#! /usr/bin/env python3

import copy
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns

from history_matching import quick_read_xl, quick_read_hdf

WORK_DIR = Path(__file__).parent.absolute()
PLOT_DIR = WORK_DIR / "Plots"


def main():

    iteration = int(re.search(r"[+-]?\d+", WORK_DIR.parts[-1]).group())
    res = 10
    sns.set_style("whitegrid", {"axes.grid": False})

    params_file = WORK_DIR.parent / "Params.xlsx"
    param_info = quick_read_xl(params_file, "Params").set_index("Name")
    Xcols = param_info.index.tolist()

    candidates = quick_read_hdf(WORK_DIR / f"Candidates_for_iter{iteration+1}.hd5", "values")

    gs = gridspec.GridSpec(len(Xcols), len(Xcols))

    cuts_dir = WORK_DIR.parent / f"iter{iteration}" / "Cuts"

    for cut_name in [
        name.name
        for name in cuts_dir.iterdir()
        if name.is_dir()
    ]:
        print(f"Working on cut {cut_name}")

        implausible_col = f"Implausible_{iteration}_{cut_name}"
        implausibility_col = f"Implausibility_{iteration}_{cut_name}"

        fig = plt.figure(figsize=(16, 12), dpi=300)
        for row in range(len(Xcols)):
            for col in range(len(Xcols)):

                if col == row:
                    ax = plt.subplot(gs[row, col])
                    ax.text(
                        0.5,
                        0.5,
                        Xcols[row].replace(" ", "\n"),
                        rotation=45,
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    ax.set_axis_off()

                elif col > row:

                    xc = Xcols[row]
                    yc = Xcols[col]

                    cc = candidates[[xc, yc, implausible_col, implausibility_col]]

                    bins = []
                    centers = []

                    for var in [xc, yc]:

                        binname = f"Binned {var}"
                        bins.append(binname)
                        edges = np.linspace(
                            param_info.loc[var, "Min"],
                            param_info.loc[var, "Max"],
                            res + 1,
                        )

                        centers.append(
                            [(a + b) / 2.0 for a, b in zip(edges[:-1], edges[1:])]
                        )

                        cc[binname] = pd.cut(
                            cc[var],
                            edges,
                            labels=centers[-1],
                            include_lowest=True,
                            right=False,
                        )

                    all_inds = pd.DataFrame(
                        index=pd.MultiIndex.from_tuples(
                            [(x, y) for x in centers[0] for y in centers[1]]
                        )
                    )

                    all_inds.index.rename(bins, level=[0, 1], inplace=True)

                    cxy_gb = cc.groupby(bins)

                    p = ( 1 - cxy_gb[implausible_col].sum() / cxy_gb[implausible_col].count() )
                    impl_frac = np.log10(p)  # /(1-p))
                    # impl_frac.name = 'Non-Implausibility Fraction'
                    impl_frac = all_inds.merge(
                        impl_frac.to_frame(),
                        left_index=True,
                        right_index=True,
                        how="left",
                    )
                    Z = impl_frac.values.reshape((res, res))

                    ax = plt.subplot(gs[row, col])
                    masked_array = np.ma.array(Z, mask=np.isnan(Z))
                    cmap = copy.copy(cm.cool_r)
                    cmap.set_bad("gray", 1.0)  # color, alpha
                    # nearest, bicubic
                    ax.imshow(
                        masked_array,
                        interpolation="nearest",
                        cmap=cmap,
                        aspect="equal",
                        origin="lower",
                    )  # , vmin=0, vmax=1
                    ax.set_axis_off()
                    ax.set_title("Fraction Implausible, log10")

                    impl_min = np.log10(cxy_gb[implausibility_col].min())
                    # impl_min.name = 'Min Implausibility'
                    impl_min = all_inds.merge(
                        impl_min.to_frame(),
                        left_index=True,
                        right_index=True,
                        how="left",
                    )
                    Z = impl_min.values.reshape((res, res))  # .transpose()

                    ax = plt.subplot(gs[col, row])
                    masked_array = np.ma.array(Z, mask=np.isnan(Z))
                    cmap = copy.copy(cm.autumn)
                    cmap.set_bad("gray", 1.0)  # color, alpha
                    # nearest, bicubic
                    ax.imshow(
                        masked_array,
                        interpolation="nearest",
                        cmap=cmap,
                        aspect="equal",
                        origin="lower",
                    )  # vmin=0, vmax=1,
                    ax.set_title("Implausibility Min, log10")
                    ax.set_axis_off()

        if not PLOT_DIR.exists(): PLOT_DIR.mkdir()
        fig_file = PLOT_DIR / f"Implausibility_{cut_name}.pdf"
        print(f"Saving figure to '{fig_file}'...")        
        fig.savefig(fig_file)
        plt.close(fig)

    return


if __name__ == "__main__":
    main()
