import os, re
from history_matching import HistoryMatchingCut

def constrain(samples):
    return [True]*samples.shape[0]
    '''
    good = samples['Male To Female Young'] >= samples['Male To Female Old']

    if 'Male To Female Young' in samples and 'Male To Female Old' in samples:
        #  --> 'Male To Female Young' >= 'Male To Female Old'
        good &= (samples['Male To Female Young'] >= samples['Male To Female Old'])

    if 'Symptomatic Linking Min' in samples and 'Symptomatic Linking Max' in samples:
        #  --> 'Symptomatic Linking Min' <= 'Symptomatic Linking Max'
        good &= (samples['Symptomatic Linking Min'] <= samples['Symptomatic Linking Max'])

    if 'PreART Linking Min' in samples and 'PreART Linking Max' in samples:
        #  --> 'PreART Linking Min' <= 'PreART Linking Max'
        good &= (samples['PreART Linking Min'] <= samples['PreART Linking Max'])

    if 'ART Linking Min' in samples and 'ART Linking Max' in samples:
        #  --> 'ART Linking Min' <= 'ART Linking Max'
        good &= (samples['ART Linking Min'] <= samples['ART Linking Max'])

    if 'Risk Ramp Min' in samples and 'Risk Ramp Max' in samples:
        #  --> 'Risk Ramp Min' <= 'Risk Ramp Max'
        good &= (samples['Risk Ramp Min'] <= samples['Risk Ramp Max'])

    return good
    '''

# History Matching!
hm = HistoryMatchingCut(
    cut_dir = 'Cuts',
    iteration = int(re.search(r'iter(\d+)', os.getcwd()).group(1))
)


### Cut #######################################################################
print("="*80, "\nCut\n", "="*80)
###############################################################################
(_, rejected_percent) = hm.cut(num_desired_candidates=25, constraint = constrain)

