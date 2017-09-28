import copy
import math
import os
import random
import sys

# dtk-tools requirements
from dtk.utils.builders.ConfigTemplate import ConfigTemplate
from dtk.utils.builders.TaggedTemplate import CampaignTemplate, DemographicsTemplate
from dtk.utils.builders.TemplateHelper import TemplateHelper
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.ModBuilder import ModBuilder, ModFn
from simtools.SetupParser import SetupParser

from history_matching.newlib.case import Case
from history_matching.newlib.sample_file import SampleFile

SetupParser.default_block = 'HPC'

# Replicates (>1)are currently untested in history matching. The underlying code may not work as expected.
N_rep_per_sample = 1

# ck4, dumb, should be in __main__ section
template_dir = os.path.join(os.path.dirname(__file__), 'Templates')
# template_dir = os.path.join('c:\\', 'Users', 'ckirkman', 'code', 'dtk-tools', 'ck4', 'examples', 'Templates')

cfg = ConfigTemplate.from_file(template_filepath=os.path.join(template_dir, 'config.json'))
# Here is how you set the tag, "__KP", for campaign, demographics, and potentially also config files
cpn = CampaignTemplate.from_file(template_filepath=os.path.join(template_dir, 'campaign.json'), tag='__KP')
# cpn_outbreak = CampaignTemplate.from_file(template_filepath=os.path.join(template_dir, 'campaign_outbreak_only.json'))
demog_pfa = DemographicsTemplate.from_file(template_filepath=os.path.join(template_dir, 'pfa_overlay.json'))

static_config_params = {
    'Base_Population_Scale_Factor':  1/10000.0
}
static_campaign_params = {
    'Intervention_Config__KP_STI_CoInfection_At_Debut.Demographic_Coverage': 0.055,
    'Demographic_Coverage__KP_Seeding_15_24_Male': 0.035
}
static_demog_params = {
    'Relationship_Parameters__KP_TRANSITORY_and_INFORMAL.Coital_Act_Rate': 0.5
}
# Once we have the static parameters dictionaries, we need to apply them to our templates
cfg.set_params(static_config_params)              # <-- Set static config parameters
cpn.set_params(static_campaign_params)            # <-- Set static campaign parameters for campaign.json
demog_pfa.set_params(static_demog_params)

templates = TemplateHelper()

# Standard DTKConfigBuilder
config_builder = DTKConfigBuilder.from_files(
    os.path.join(template_dir, 'config.json'),
    os.path.join(template_dir, 'campaign.json')
)
config_builder.ignore_missing = True
config_builder.set_exe_collection('EMOD 2.10')

def map_sample_to_model_input(config_builder, sample_id, replicate_idx, params, table_base, sample):
    print sample

    table = copy.deepcopy(table_base)
    table['TAGS'].update({
        '__sample_id__': sample_id,
        '__replicate_index__': replicate_idx
    })
    table['Run_Number'] = random.randint(0, 1e6)

    # This section is intended to provide non-passthrough (renaming) mappings from
    # Params.xlsx entries to model configuration parameters.

    if 'LOG Acute Infectiousness' in sample:
        value = sample.pop('LOG Acute Infectiousness')
        table['Typhoid_Acute_Infectiousness'] = math.exp(value)
    if 'LOG Immunity Duration' in sample:
        value = sample.pop('LOG Immunity Duration')
        table['Typhoid_Immunity_Memory'] = math.exp(value)

    if 'LOG Contact Exposure Period' in sample:
        value = sample.pop('LOG Contact Exposure Period')
        table['Typhoid_Contact_Exposure_Rate'] = 1.0 / math.exp(value)

    if 'LOG Environmental Exposure Period' in sample:
        value = sample.pop('LOG Environmental Exposure Period')
        table['Typhoid_Environmental_Exposure_Rate'] = 1.0 / math.exp(value)

    if 'LOG Long-cycle reservoir decay period' in sample:
        value = sample.pop('LOG Long-cycle reservoir decay period')
        table['Node_Contagion_Decay_Rate'] = 1.0 / math.exp(value)

    if 'Exposure Age Median' in sample:
        value = sample.pop('Exposure Age Median')
        table['Typhoid_Exposure_Lambda'] = 20.0 / value - 2.0

    # Handle passthrough (simple renaming) mappings from Params.xlsx entries to model configuration parameters.
    for param_name, p in params.iterrows():
        if param_name in sample and 'MapTo' in p:
            if isinstance(p['MapTo'], float) and math.isnan(p['MapTo']):
                continue
            table[p['MapTo']] = sample.pop(param_name)

    # 'id' is not a parameter that needs varying. This line should exist in all commission parameter
    # mapping scripts.
    if sample.get('id', None) is not None:
        sample.pop('id')

    for name, value in sample.iteritems():
        print 'UNUSED PARAMETER:', name
    assert (len(sample) == 0)  # All params used

    return templates.mod_dynamic_parameters(config_builder, table)


if __name__ == "__main__":

    # First, handle arguments to this script
    if len(sys.argv) != 4:
        print('Usage: %s <case_dir> <iter_number> <data_source>' % (os.path.basename(__file__)))
        exit()

    case_dir = os.path.abspath(sys.argv[1])
    iter_number = int(sys.argv[2])
    data_source = sys.argv[3]

    # Initialize the history_matching module objects to represent the relevant commissioning data
    case = Case(case_dir)
    parameters = case.parameters
    iteration = case.get_iteration(iteration_number=iter_number)
    data_source = iteration.get_data_source(source=data_source)

    samples = SampleFile(data_source.samples_filename).samples
    if samples is None:
        raise Exception('No samples exist in iteration %d data source: %s, cannot commission simulations.'
                        % (iteration.iteration_number, data_source.name))

    # Build up the dtk/emod simulation configuration variations

    table_base = {
        'ACTIVE_TEMPLATES': [cfg, cpn, demog_pfa],
        'TAGS': {'BayesianHistoryMatching': None, 'Iteration': iter_number}
    }

    exp_builder = ModBuilder.from_combos(
        [
            ModFn(map_sample_to_model_input,
                  sample[0],  # <-- sample index
                  rep,  # <-- replicate index
                  parameters,
                  table_base,
                  {k: v for k, v in zip(samples.columns.values, sample[1:])})
            for sample in samples.itertuples() for rep in range(N_rep_per_sample)
        ])

    print('Ready to run simulations.')

    # Gotta do this before we try to build sims to commission
    SetupParser.init()

    # Run the simulations
    em = ExperimentManagerFactory.init()
    em.run_simulations(config_builder=config_builder,
                       exp_builder=exp_builder,
                       exp_name='hm-commission-example-iter%d'%iteration.iteration_number)

