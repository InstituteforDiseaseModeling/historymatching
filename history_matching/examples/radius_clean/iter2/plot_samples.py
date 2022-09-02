#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from history_matching import quick_read_xl

WORK_DIR = Path(__file__).parent.absolute()
PLOT_DIR = WORK_DIR / "Plots"
if not PLOT_DIR.exists(): PLOT_DIR.mkdir()
DATA_DIR = sorted([dir for dir in WORK_DIR.glob("Data*") if dir.is_dir()])[0]


def main(data_dir = None):

    if data_dir is None: data_dir = DATA_DIR

    samples = quick_read_xl(data_dir / "Samples.xlsx", "Samples").set_index("Sample")
    params = quick_read_xl(data_dir / "Samples.xlsx", "Params").set_index("Name")

    s = samples.stack()
    s.name = "Value"
    s.index.rename(names="Variable", level=1, inplace="True")
    s = s.reset_index().drop("Sample", axis=1).sort_values("Variable").reset_index()


    def plt_hist(**kwargs):
        data = kwargs.pop("data")
        sns.displot(data["Value"])
        param_name = data["Variable"].iloc[0]
        plt.xlim((params.loc[param_name, "Min"], params.loc[param_name, "Max"]))


    def plt_sample(**kwargs):
        data = kwargs.pop("data")
        param_name = data["Variable"].iloc[0]
        pt = kwargs["pt"]
        plt.plot(pt[param_name], 0, "r^", lw=2, ms=10)


    g = sns.FacetGrid(
        s.reset_index(),
        col="Variable",
        col_wrap=5,
        sharex=False,
        sharey=False,
        size=3,
        aspect=1,
        palette=None,
        row_order=None,
        col_order=None,
        hue_order=None,
        hue_kws=None,
        dropna=True,
        legend_out=True,
        despine=True,
        margin_titles=False,
        xlim=None,
        ylim=None,
        subplot_kws=None,
        gridspec_kws=None,
    )
    print("Mapping")
    g = g.map_dataframe(plt_hist)

    # SHOW THE LOCATION OF ONE POINT:
    # g = g.map_dataframe(plt_sample, pt = samples.loc[4408])

    g = g.set_titles("{col_name}")
    fig_file = PLOT_DIR / "Samples.pdf"
    print(f"Saving figure to '{fig_file}'")
    plt.savefig(fig_file)

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=Path, default=None)

    args = parser.parse_args()

    main(args.data_dir)
