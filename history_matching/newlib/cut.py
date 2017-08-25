import json
import os
import numpy as np
import pandas as pd

from newlib.basis import Basis

class Cut(object):
    GLM_BASIS_FILENAME = 'basis_glm.json'
    GPR_BASIS_FILENAME = 'basis_gpr.json'
    
    def __init__(self, full_path_directory):
        self.directory = full_path_directory
        self.name = os.path.basename(full_path_directory)

        # load GLM basis if it exists
        self.glm_basis_filename = os.path.join(self.directory, self.GLM_BASIS_FILENAME)
        self.glm_basis = self._load_basis(self.glm_basis_filename)
        
        # load GPR basis if it exists
        self.gpr_basis_filename = os.path.join(self.directory, self.GPR_BASIS_FILENAME)
        self.gpr_basis = self._load_basis(self.gpr_basis_filename)
            
    def _load_basis(self, basis_filename):
        if os.path.exists(basis_filename):
            with open(basis_filename) as data_file:
                config = json.load( data_file )
                basis = Basis.deserialize(config['Basis'])
        else:
            basis = None
        return basis

    # ck4, this should be combined with _load_basis() above into a Basis class, but I am in a hurry right now.
    # Extend Dan's Basis class, or a new one?
    def _load_fitted_values(self, basis_filename):
        if os.path.exists(basis_filename):
            with open(basis_filename) as data_file:
                config = json.load( data_file )
                fitted_values = pd.read_json(config['Fitted_Values'], orient='split').set_index(['Sample_Id', 'Exp_Id', 'id', 'Sim_Id']).squeeze()
        else:
            fitted_values = None
        return fitted_values
        
    
    def make_bases(self, param_info, inputs, results, remake='none'):
        if param_info is None:
            raise Exception('parameter information must be provided')

        # delete any bases that were specified for remaking
        to_remove = []
        if remake == 'all': # ck4, 'all', 'none', 'gpr' should be class constants on a Basis class... someday
            to_remove.append(self.glm_basis_filename)
            to_remove.append(self.gpr_basis_filename)
        elif remake == 'gpr':
            to_remove.append(self.gpr_basis_filename)
        for filename in to_remove:
            try:
                os.remove(filename)
            except OSError:
                pass
        param_names = param_info.index.tolist()
        print 'All available parameters:'
        print ' *','\n * '.join(param_names)
                
        # now create bases

        # GLM basis
        if not os.path.exists(self.glm_basis_filename):
            basis_glm = Basis.polynomial_basis(params=param_names, intercept = True, first_order=True, second_order=True, third_order=False, param_info=param_info)
            basis_glm.plot_regularize(inputs, results, alpha = np.logspace(-3,0, 25), scaleX=True)
            alpha_glm = float(raw_input('What would you like to use for the GLM regularization parameter, alpha_glm = '))
            fitted_values = basis_glm.regularize(inputs, results, alpha = alpha_glm, scaleX=True) # 100 for thrid_order
            print 'Regularization for GLM selected:\n', ' *','\n * '.join(basis_glm.get_terms())

            basis_dir = os.path.dirname(self.glm_basis_filename)
            if not os.path.exists(basis_dir):
                os.makedirs(basis_dir)

            with open(self.glm_basis_filename, 'w') as fout:
                json.dump( {
                    'Basis': basis_glm.serialize(),
                    'Fitted_Values': fitted_values.reset_index().to_json(orient='split')
                }, fout, indent=4)
                print 'fitted value: %s' % fitted_values.to_json(orient='split')
        else:
            print('Not creating GLM basis, as it already exists and we were not instructed to force recreation.')

        # ck4, need to set fitted_values either way (for use in GPR, below)... figure out how by looking at original bhm.py
        self.glm_basis = self._load_basis(self.glm_basis_filename)
        fitted_values = self._load_fitted_values(self.glm_basis_filename)
        
        # GPR basis
        if not os.path.exists(self.gpr_basis_filename):
            basis_gpr = Basis.polynomial_basis(params=param_names, intercept = False, first_order=True, param_info=param_info)
            results_err = results - fitted_values
            basis_gpr.plot_regularize(inputs, results_err, alpha = np.logspace(-6, 0, 25), scaleX=True)
            alpha_gpr = float(raw_input('What would you like to use for the GPR regularization parameter, alpha_gpr = '))
            basis_gpr.regularize(inputs, results_err, alpha = alpha_gpr, scaleX=True)
            print 'Regularization for GPR selected:\n', ' *','\n * '.join(basis_gpr.get_terms())
            
            basis_dir = os.path.dirname(self.glm_basis_filename)
            if not os.path.exists(basis_dir):
                os.makedirs(basis_dir)

            with open(self.gpr_basis_filename, 'w') as fout:
                json.dump( { 'Basis': basis_gpr.serialize(), }, fout, indent=4)
        else:
            print('Not creating GPR basis, as it already exists and we were not instructed to force recreation.')
            
        # now set instance members
        self.gpr_basis = self._load_basis(self.gpr_basis_filename)
