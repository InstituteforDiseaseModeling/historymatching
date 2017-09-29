import argparse
import re

import commands_args
from history_matching.newlib.case import Case # fix up all newlib references once things are packaged nicely
from history_matching.newlib.sample_file import SampleFile

CONSTRAINT_METHOD = 'constrain'

# # ck4, constraints need testing
def sample(args):
    # try to load and error check any provide constraint information
    if args.constraint_filename:
        import simtools.Utilities.Initialization as init  # from dtk tools
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

    # determine how many and where to obtain sample parameter points from
    n_samples = args.n_samples

    case = Case(case_directory=args.case_directory)
    iteration = case.get_iteration(args.iteration_number)
    if args.sample_file:
        samples = SampleFile(args.sample_file).samples
    else:
        samples = case.generate_samples(n_samples=n_samples,
                                        iteration_number=iteration.iteration_number,
                                        constraint=constraint)

    # apply and write samples to the selected iteration/data source
    iteration.set_samples(samples=samples)
    iteration.write_samples(data_source_name=args.data_source)

def fit(args):
    case = Case(case_directory=args.case_directory)
    iteration = case.get_iteration(args.iteration_number)

    remake_basis = args.remake_basis.lower()
    allowed_values = ['all', 'none', 'gpr']
    if remake_basis not in allowed_values:
        raise Exception('--remake-basis must be one of (case insensitive): %s' % allowed_values)

    # load data sources csv
    data_sources = case.load_data_sources_csv(filename=args.data_sources)

    # verify that the requested comparison field is in both the reference data.
    if args.field not in case.reference_data.fields:
        raise Exception('Selected field is not in the reference data: %s' % args.field)

    # ck4, compress field name/value/stddev into a simple class and pass as one arg to iteration.fit
    field = args.field
    target = case.reference_data.value(field=field)
    target_std = case.reference_data.stddev(field=field)

    iteration.fit(cut_name           = args.cut_name,
                  data_sources       = data_sources,
                  field              = field,
                  target             = target,
                  target_std         = target_std,
                  force_optimize_glm = args.optimize_glm,
                  force_optimize_gpr = args.optimize_gpr,
                  implausibility_threshold = args.implausibility_threshold,
                  remake_basis   = remake_basis)

def main():
    parser = argparse.ArgumentParser(prog='hm')
    subparsers = parser.add_subparsers()
    
    # 'hm sample'
    commands_args.populate_sample(subparsers, function=sample)

    # 'hm fit'
    commands_args.populate_fit(subparsers, function=fit)
    
    # parse args for and run specified command
    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__':
    main()
