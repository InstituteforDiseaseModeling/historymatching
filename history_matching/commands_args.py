def populate_set_samples(subparsers, function):
    parser = subparsers.add_parser('set-samples', help='')
    parser.add_argument('-c', '--case', dest='case_directory', type=str, required=True,
                        help='BHM case directory. Required.')
    parser.add_argument('-i', '--iteration', dest='iteration_number', type=int, required=True,
                        help='Current iteration number. Required.')
    parser.add_argument('-s', '--sample-file', dest='sample_file', type=str, default=None,
                        help='A file containing samples to use.')
    parser.add_argument('-n', '--number', dest='n_samples', type=int, default=None,
                        help='Number of parameter space points to sample in this iteration.')
    parser.set_defaults(func=function)

def populate_fit(subparsers, function):
    parser = subparsers.add_parser('fit', help='')
    parser.add_argument('-c', '--case', dest='case_directory', type=str, required=True,
                        help='BHM case directory. Required.')
    parser.add_argument('-i', '--iteration', dest='iteration_number', type=int, required=True,
                        help='Current iteration number. Required.')
    parser.add_argument('-C', '--cut-name', dest='cut_name', type=str, required=True,
                        help='Name of cut to use. Required.')

    # ck4, populate this on a per-directory basis from the inputs csv file
    # parser.add_argument('-t', '--training-fraction', dest='training_fraction', type=float, default=0.75,
    #                     help='Fraction of the training data directory to use in training (0-1) (Default: 0.75).')

    parser.add_argument('--no-optimize-glm', dest='optimize_glm', action='store_false', default=True,
                        help='Do not force optimization of the GLM (Default: force).')
    parser.add_argument('--no-optimize-gpr', dest='optimize_gpr', action='store_false', default=True,
                        help='Do not force optimization of the GPR (Default: force).')

    # ck4, populate these with new input csv file
    parser.add_argument('--data-sources', dest='data_sources', type=str, required=True,
                        help='A CSV file detailing which directories and their configurations to use in training and'
                             'testing GLM and GPR. Required.')
    # parser.add_argument('--training-dir', dest='training_directory', type=str, required=True,
    #                     help='Training iteration directory to use. Required.')
    # parser.add_argument('-d', '--data-dirs', dest='data_directories', nargs='+', type=str, default=[],
    #                     help='Non-training iteration directories to use for history matching inputs.')

    # ck4,  move this into Case object data?
    parser.add_argument('--implausibility_threshold', dest='implausibility_threshold', type=float, default=3,
                        help='The minimum sigma difference between data and fitted value to determine implausibility'
                             '(Default: 3).')

    parser.add_argument('--remake-basis', dest='remake_basis', type=str, default="NONE",
                        help='Toggle basis recreation mode before fitting: GPR, ALL (GLM and GPR), or NONE')

    # ck4, combine into one command line arg, something like: --target 25:4
    # ck4, or should --target-std info be part of the Case (and invariant between Iterations)
    parser.add_argument('--target', dest='target', type=float, required=True,
                        help='Desired value for parameter space points to attempt to match. Required.')
    parser.add_argument('--target-std', dest='target_std', type=float, required=True,
                        help='Desired value standard deviation. Same units as --target . Required.')
    parser.set_defaults(func=function)

def populate_cut_params(subparsers, function):
    parser = subparsers.add_parser('cut', help='')
    parser.add_argument('-c', '--case', dest='case_directory', type=str, required=True,
                        help='BHM case directory. Required.')
    parser.add_argument('-i', '--iteration', dest='iteration_number', type=int, required=True,
                        help='Current iteration number. Required.')
    parser.add_argument('-n', '--number', dest='n_candidates', type=int, required=True,
                        help='Keep at least and close to this many candidate points in the cut.')

    # ck4, remove this argument and make the constraint file a well-known filename in the Case; use if present!
    parser.add_argument('-C', '--constraint-file', dest='constraint_filename', type=str,
                        help='Python file containing the top-level method \'constrain\' to be used for constraining'
                             'sample space candidates.')
    parser.set_defaults(func=function)
