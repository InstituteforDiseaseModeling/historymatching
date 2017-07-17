import argparse
import commands_args

from newlib.iteration import Iteration # fix up all newlib references once things are packaged nicely
from newlib.sample_file import SampleFile

def set_samples(args):
    if args.n_samples:
        n_samples = args.n_samples
        if not args.parameter_file:
            raise Exception('Must provide parameter file via argument --params when generating parameter space points via -n .')
    else:
        n_samples = None

    iteration = Iteration(args.iteration_directory, parameter_filename=args.parameter_file)
    if args.previous_iteration_directory:
        if args.previous_iteration_directory.lower() == Iteration.NONE:
            previous_iteration = None # allowing for e.g. iteration 0 where there is no previous iteration
        else:
            previous_iteration = Iteration(args.previous_iteration_directory)
    else:
        previous_iteration = None
    
    if args.sample_file:
        sample_file = SampleFile(args.sample_file)
    else:
        sample_file = None

    iteration.set_samples(previous_iteration=previous_iteration,
                          n_samples=n_samples,
                          samples_file = sample_file)
    iteration.write_samples()

# ck4, make this method nice; pasted in from cut.py currently.
# ck4, needs a run through test
def cut_parameter_space(args):
    import os, re
    
    # Example constraint function:
    #def day_sum(row):
    #    return row[['Env Ramp Up', 'Env Ramp Down', 'Env Cutoff']].sum() < 365
    
    
    iteration = Iteration(args.iteration_directory, parameter_filename=args.parameter_file)
    iteration.cut_param_space(n_desired_candidates=args.n_candidates, constraint = None) # ck4, constraints need to be added back somehow

#def make_bases(args):
#    iteration = Iteration(args.iteration_directory, parameter_filename=args.parameter_file)
#    iteration.make_bases(args.cut_name, force = args.force)

def fit(args):
    if args.training_fraction <= 0 or args.training_fraction >= 1:
        raise Exception('Training fraction must be greater than 0 and less than 1.')
    
    iteration = Iteration(args.iteration_directory, parameter_filename=args.parameter_file)
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
