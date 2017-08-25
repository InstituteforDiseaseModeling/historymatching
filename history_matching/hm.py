import argparse

import commands_args
from newlib.case import Case # fix up all newlib references once things are packaged nicely
from newlib.sample_file import SampleFile

CONSTRAINT_METHOD = 'constrain'

def sample(args):
    if args.n_samples:
        n_samples = args.n_samples
    else:
        n_samples = None

    case = Case(case_directory=args.case_directory)
    iteration = case.get_iteration(args.iteration_number)
    previous_iteration = case.get_previous_iteration(args.iteration_number)
    if args.sample_file:
        sample_file = SampleFile(args.sample_file)
    else:
        sample_file = None

    iteration.set_samples(previous_iteration=previous_iteration,
                          n_samples=n_samples,
                          samples_file = sample_file)
    iteration.write_samples(data_source_name=args.data_source)

# from cut.py
# ck4, needs a run through test, especially constraints
def cut_parameter_space(args):
    # try to load and error check any provide constraint information
    if args.constraint_filename:
        import simtools.Utilities.Initialization as init # from dtk tools
        args.config_name = args.constraint_filename
        mod = init.load_config(args)
        constraint_method = CONSTRAINT_METHOD
        try:
            constraint = getattr(mod, constraint_method)
        except AttributeError as e:
            raise AttributeError('Error in loading constraint method: %s from module: %s' %
                                 (constraint_method, args.constraint_filename))
    else:
        constraint = None

    case = Case(case_directory=args.case_directory)
    case.cut_param_space(iteration_number=args.iteration_number,
                         n_desired_candidates=args.n_candidates,
                         constraint = constraint)

# ck4, this method and code it calls needs to properly handle and recognize the data from the new csv
# data format for training/data directories and per directory training_fraction/GLM/GPR usage.
def fit(args):
    # ck4, use args.training_sources : a csv filename, instead of data_directories/training_directory, training_fraction
    # ... and to determine internally to iteration.fit() which to use for GPR and/or GLM.
    case = Case(case_directory=args.case_directory)
    iteration = case.get_iteration(args.iteration_number)
    
    remake_basis = args.remake_basis.lower()
    allowed_values = ['all', 'none', 'gpr']
    if remake_basis not in allowed_values:
        raise Exception('--remake-basis must be one of (case insensitive): %s' % allowed_values)

    # load data sources csv
    data_sources = case.load_data_sources_csv(filename=args.data_sources)
    iteration.fit(cut_name           = args.cut_name,
                  data_sources       = data_sources,
                  target             = args.target,
                  target_std         = args.target_std,
                  force_optimize_glm = args.optimize_glm,
                  force_optimize_gpr = args.optimize_gpr,
                  implausibility_threshold = args.implausibility_threshold,
                  remake_basis   = remake_basis)

def main():
    parser = argparse.ArgumentParser(prog='hm')
    subparsers = parser.add_subparsers()
    
    # 'hm set-samples'
    commands_args.populate_sample(subparsers, function=sample)
    
    # 'hm cut-params'
    commands_args.populate_cut_params(subparsers, function=cut_parameter_space)

    # 'hm fit'
    commands_args.populate_fit(subparsers, function=fit)
    
    # parse args for and run specified command
    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__':
    main()
