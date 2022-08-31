#! /usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from history_matching import quick_read_xl

WORK_DIR = Path(__file__).parent.absolute()


def main():

    exp_dirs = [dir for dir in WORK_DIR.glob("Data_*") if dir.is_dir()]

    if len(exp_dirs) == 0:
        print("Did not find any folders with name Data_*.  Please generate samples first.")
        exit()

    for exp_dir in exp_dirs:

        samples = quick_read_xl( exp_dir / "Samples.xlsx", "Samples" ).set_index("Sample")
        params = quick_read_xl( exp_dir / "Samples.xlsx", "Params" ).set_index("Name")

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
            col_wrap=min(params.shape[0], 5),
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
        fig_file = exp_dir / "Samples.pdf"
        print(f"Saving figure to '{fig_file}'...")
        plt.savefig(fig_file)

    return


if __name__ == "__main__":
    main()
