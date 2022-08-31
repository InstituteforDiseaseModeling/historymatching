#! /usr/bin/env python3

from argparse import ArgumentParser
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from history_matching.HistoryMatching import HistoryMatching
from history_matching.quick_read import quick_read_xl
from history_matching.basis import Basis

WORK_DIR = Path(__file__).parent.absolute()


def main(alpha_glm=0.001, alpha_gpr=0.0, force_optimize_glm=True, force_optimize_gpr=True):

    print(f"Using alpha_glm={alpha_glm}")
    print(f"Using alpha_gpr={alpha_gpr}")
    print(f"Using force_optimize_glm={force_optimize_glm}")
    print(f"Using force_optimize_gpr={force_optimize_gpr}")

    iteration = int(re.search(r"[+-]?\d+", WORK_DIR.parts[-1]).group())
    training_fraction = 0.75
    implausibility_threshold = 3

    cut_name = "RadiusShouldBe15"
    desired_result = 15
    discrepancy_std = 0.1 * desired_result
    print(f"Desired result is {desired_result}")

    # Data
    params_file = WORK_DIR.parent / "Params.xlsx"
    samples_fn = "Samples.xlsx"
    results_fn = "Results.xlsx"

    sim_inputs = []
    sim_results = []
    for idx, exp_path in enumerate(WORK_DIR.glob("Data_*")):

        read = quick_read_xl(exp_path / samples_fn, "Samples")
        exp_id = exp_path.parts[-1]
        read["Exp_Id"] = exp_id
        read["Sample_Id"] = read["Sample"].apply(lambda x: f"{exp_id}.{x:06}")
        read = read.set_index("Sample_Id").sort_index()

        # Train/test split
        if idx == 0:

            read["Train"] = False
            nSamp = len(read.index.get_level_values("Sample_Id"))
            nTrain = int(round(training_fraction * nSamp))
            read.iloc[:nTrain, read.columns == "Train"] = True

        else:
            read["Train"] = True

        sim_inputs.append(read)

        read = quick_read_xl(exp_path / results_fn, "Sheet1")
        read["Exp_Id"] = exp_id
        read["Sample_Id"] = read["Sample"].apply(lambda x: f"{exp_id}.{x:06}")
        sim_results.append(read.set_index("Sample_Id").sort_index())

    inputs = pd.concat(sim_inputs)
    sim_results_all = pd.concat(sim_results)

    print(np.sqrt(inputs["First Parameter"] ** 2 + inputs["Second Parameter"] ** 2))
    print(sim_results_all["Sim_Result"])

    sim_results_all.set_index(["Sim_Id"], append=True, inplace=True)
    results = sim_results_all["Sim_Result"]

    cuts_dir = WORK_DIR / "Cuts" / cut_name
    if not cuts_dir.exists(): cuts_dir.mkdir(parents=True)

    figs_dir = WORK_DIR / "Plots"
    if not figs_dir.exists(): figs_dir.mkdir(parents=True)

    param_info = quick_read_xl(params_file, "Params").set_index("Name")
    param_names = param_info.index.tolist()
    print("All available parameters:")
    print(" *", "\n * ".join(param_names))

    # Choose GLM inputs
    try:
        with (cuts_dir / "basis_glm.json").open("r") as data_file:
            config = json.load(data_file)
            basis_glm = Basis.deserialize(config["Basis"])
            fitted_values = (
                pd.read_json(config["Fitted_Values"], orient="split")
                .set_index(["Sample_Id", "Sim_Id"])
                .squeeze()
            )
    except:
        basis_glm = Basis.polynomial_basis(
            params=param_names,
            intercept=True,
            first_order=True,
            second_order=True,
            third_order=False,
            param_info=param_info,
        )

        basis_glm.plot_regularize(
            inputs, results, alpha=np.logspace(-3, 1, 25), scaleX=True, fig_file = figs_dir / "glm_basis.png"
        )
        # alpha_glm = float(
        #     input(
        #         "What would you like to use for the GLM regularization parameter, alpha_glm = [0.001] "
        #     )
        # )
        # alpha_glm = 1e-3

        fitted_values = basis_glm.regularize(
            inputs, results, alpha=alpha_glm, scaleX=True
        )  # 100 for third_order

        print(type(basis_glm.get_terms()))
        print(
            "Regularization for GLM selected:\n * " + "\n * ".join(basis_glm.get_terms())
        )
        with (cuts_dir / "basis_glm.json").open("w") as fout:
            json.dump(
                {
                    "Basis": basis_glm.serialize(),
                    "Fitted_Values": fitted_values.reset_index().to_json(
                        orient="split"
                    ),
                },
                fout,
                indent=4,
            )

    # Choose GPR inputs
    try:
        with (cuts_dir / "basis_gpr.json").open("r") as data_file:
            config = json.load(data_file)
            basis_gpr = Basis.deserialize(config["Basis"])
    except:
        basis_gpr = Basis.polynomial_basis(
            params=param_names, intercept=False, first_order=True, param_info=param_info
        )
        results_err = results - fitted_values

        basis_gpr.plot_regularize(
            inputs, results_err, alpha=np.logspace(-3, 1, 25), scaleX=True, fig_file = figs_dir / "gpr_basis.png"
        )
        # alpha_gpr = float(
        #     input(
        #         "What would you like to use for the GPR regularization parameter, alpha_gpr = [0] "
        #     )
        # )

        basis_gpr.regularize(inputs, results_err, alpha=alpha_gpr, scaleX=True)
        print(
            "Regularization for GPR selected:\n * " + "\n * ".join(basis_gpr.get_terms())
        )
        with (cuts_dir / "basis_gpr.json").open("w") as fout:
            json.dump(
                {
                    "Basis": basis_gpr.serialize(),
                },
                fout,
                indent=4,
            )

    # History Matching!
    hm = HistoryMatching(
        cut_name=cut_name,
        param_info=param_info,
        inputs=inputs,
        results=results,
        desired_result=desired_result,
        iteration=iteration,
        implausibility_threshold=implausibility_threshold,
        discrepancy_var=discrepancy_std**2,
        training_fraction=training_fraction,
        iterdir = WORK_DIR
    )
    hm.save()

    # If desired, you can filter train/test/both data with lower and upper bounds on the result
    # hm.filter_data(source='Both', lower=0)

    ### GLM ###############################################################
    print(f"{'='*80}\nGeneralized Linear Modeling\n{'='*80}")
    #######################################################################
    hm.glm(
        basis=basis_glm,
        family="Gaussian",
        force_optimize_glm=force_optimize_glm,
        glm_fit_maxiter=100000,
        plot=force_optimize_glm,
        plot_data=True
    )

    ### GPR ###############################################################
    print(f"{'='*80}\nGaussian Process Regression\n{'='*80}")
    #######################################################################
    hm.gpr(
        basis=basis_gpr,
        force_optimize_gpr=force_optimize_gpr,
        sigma2_f_guess=1,
        sigma2_f_bounds=(0.1, 100),
        sigma2_n_guess=1,
        sigma2_n_bounds=(0.001, 100),
        # lengthscale_guess = [0.04313128, 0.2, 0.14240553, 0.01418867, 0.2, 0.17683428],
        lengthscale_guess=0.15,
        lengthscale_bounds=(0.001, 0.2),
        verbose=True,
        optimizer_options={
            "eps": 5e-3,
            "disp": True,
            "maxiter": 15000,
            #'ftol': 1e-1,
            #'gtol': 1e-1,
            #'factr': 1e12 # <-- Not working?
        },
        optimize_sigma2_n=True,
        log_transform=True,
        plot=True,  # force_optimize_gpr,
        plot_data=True
    )

    ### Implausibility ############################################################
    print(f"{'='*80}\nImplausibility\n{'='*80}")
    ###############################################################################
    hm.calc_and_plot_implausibility(
        plot=True, do_plot_data=True, plot_data_highlight=pd.DataFrame()
    )  # plot_data_highlight=hm.training_data.loc['8c7e4af7-1120-e711-9400-f0921c16849c.003328']

    hm.training_data.to_excel(cuts_dir / "train_data.xlsx")
    hm.test_data.to_excel(cuts_dir / "test_data.xlsx")

    print("Good")

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-l", "--alpha-glm", type=float, default=0.001, help="GLM regularization parameter [0.001]")
    parser.add_argument("-p", "--alpha-gpr", type=float, default=0.0, help="GPR regularization parameter")
    parser.add_argument("--no-force-optimize-glm", action="store_false", dest="force_optimize_glm")
    parser.add_argument("--no-force-optimize-gpr", action="store_false", dest="force_optimize_gpr")

    args = parser.parse_args()

    main(args.alpha_glm, args.alpha_gpr, args.force_optimize_glm, args.force_optimize_gpr)
