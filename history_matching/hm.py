import argparse
import commands_args

from newlib.case import Case # fix up all newlib references once things are packaged nicely
from newlib.sample_file import SampleFile

CONSTRAINT_METHOD = 'constrain'

def set_samples(args):
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
    iteration.write_samples()

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

def fit(args):
    case = Case(case_directory=args.case_directory)
    iteration = case.get_iteration(args.iteration_number)
    if args.training_fraction <= 0 or args.training_fraction >= 1:
        raise Exception('Training fraction must be greater than 0 and less than 1.')

    iteration.fit(cut_name           = args.cut_name,
                  training_directory = args.training_directory,
                  data_directories   = args.data_directories,
                  target             = args.target,
                  target_std         = args.target_std,
                  training_fraction  = args.training_fraction,
                  force_optimize_glm = args.optimize_glm,
                  force_optimize_gpr = args.optimize_gpr,
                  implausibility_threshold = args.implausibility_threshold,
                  remake_bases       = args.remake_bases)

def main():
    parser = argparse.ArgumentParser(prog='hm')
    subparsers = parser.add_subparsers()
    
    # 'hm set-samples'
    commands_args.populate_set_samples(subparsers, function=set_samples)
    
    # 'hm cut-params'
    commands_args.populate_cut_params(subparsers, function=cut_parameter_space)
    
#    # 'hm make-bases'
#    commands_args.populate_make_bases(subparsers, function=make_bases)
    
    # 'hm fit'
    commands_args.populate_fit(subparsers, function=fit)
    
    # parse args for and run specified command
    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__':
    main()
