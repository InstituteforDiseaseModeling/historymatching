def populate_set_samples(subparsers, function):
    parser = subparsers.add_parser('set-samples', help='')
    parser.add_argument('-i', '--iteration', dest='iteration_directory', type=str, required=True,
                        help='Directory for current iteration. Required.')
    parser.add_argument('-s', '--sample-file', dest='sample_file', type=str, default=None,
                        help='A file containing samples to use.')
    parser.add_argument('-n', '--number', dest='n_samples', type=int, default=None,
                        help='Number of parameter space points to sample in this iteration.')
    parser.add_argument('-p', '--previous', dest='previous_iteration_directory', type=str, default=None,
                        help='Directory for previous iteration.')
    parser.add_argument('--params', dest='parameter_file', type=str, default=None,
                        help='Parameter description .xlsx file. Required if using -n.')
    parser.set_defaults(func=function)

#     parser.add_argument(dest='script_name', default=None, help='Name of python script for custom running of simulation.')

def populate_cut_params(subparsers, function):
    parser = subparsers.add_parser('cut-params', help='')
    parser.add_argument('-i', '--iteration', dest='iteration_directory', type=str, required=True,
                        help='Directory for current iteration. Required.')
    parser.add_argument('-n', '--number', dest='n_candidates', type=int, required=True,
                        help='Keep at least and close to this many candidate points in the cut.')
    parser.add_argument('--params', dest='parameter_file', type=str, required=True,
                        help='Parameter description .xlsx file. Required.')
    parser.set_defaults(func=function)

def populate_run_samples(subparsers, function):
    parser = subparsers.add_parser('run-samples', help='')
    parser.add_argument()
    parser.set_defaults(func=function)

#def populate_make_bases(subparsers, function):
#    parser = subparsers.add_parser('make-bases', help='')
#    parser.add_argument('-i', '--iteration', dest='iteration_directory', type=str, required=True,
#                        help='Directory for current iteration. Required.')
#    parser.add_argument('-c', '--cut-name', dest='cut_name', type=str, required=True,
#                        help='Name for the cut to make. Required.')
#    parser.add_argument('--params', dest='parameter_file', type=str, required=True,
#                        help='Parameter description .xlsx file. Required.')
#    parser.add_argument('-f', '--force', dest='force', action='store_true', default=False,
#                        help='Delete existing bases and recreate it (Default: False)')
#    parser.set_defaults(func=function)

def populate_fit(subparsers, function):
    parser = subparsers.add_parser('fit', help='')
    parser.add_argument('-i', '--iteration', dest='iteration_directory', type=str, required=True,
                        help='Directory for current iteration. Required.')
    parser.add_argument('-c', '--cut-name', dest='cut_name', type=str, required=True,
                        help='Name of cut to use. Required.')
    parser.add_argument('-t', '--training-fraction', dest='training_fraction', type=float, default=0.75,
                        help='Fraction of the training data directory to use in training (0-1) (Default: 0.75).')
    parser.add_argument('--optimize-glm', dest='optimize_glm', action='store_true', default=False,
                        help='Force optimization of the GLM (Default: False).')
    parser.add_argument('--optimize-gpr', dest='optimize_gpr', action='store_true', default=False,
                        help='Force optimization of the GPR (Default: False).')
    parser.add_argument('--training-dir', dest='training_directory', type=str, required=True,
                        help='Training iteration directory to use. Required.')
    parser.add_argument('-d', '--data-dirs', dest='data_directories', nargs='+', type=str, default=[],
                        help='Non-training iteration directories to use for history matching inputs.')
    parser.add_argument('--params', dest='parameter_file', type=str, required=True,
                        help='Parameter description .xlsx file. Required.')
    parser.add_argument('--implausibility_threshold', dest='implausibility_threshold', type=float, default=3,
                        help='The minimum sigma difference between data and fitted value to determine implausibility (Default: 3).')
    parser.add_argument('--remake-bases', dest='remake_bases', action='store_true', default=False,
                        help='Remove and recreate the GLM and GPR basis before fitting (Default: False).')

    parser.add_argument('--target', dest='target', type=float, required=True,
                        help='Desired value for parameter space points to attempt to match. Required.')
    parser.add_argument('--target-std', dest='target_std', type=float, required=True,
                        help='Desired value standard deviation. Same units as --target . Required.')

    parser.set_defaults(func=function)
###
    
#     parser_run.add_argument('-q', '--quiet', action='store_true', help='Runs quietly.')
