import pandas as pd
import hashlib
import numpy as np
import json
import os
class DataManager(object):

    def __init__(self, samples_fn, results_fn, sheetname,
            training_fraction = 0.75,
            remove_zeros = False
        ):

        self.samples_fn = samples_fn
        self.results_fn = results_fn
        self.training_fraction = training_fraction
        self.remove_zeros = remove_zeros

        self.load(sheetname)


    @staticmethod
    def md5(fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


    def load(self, sheetname):

        samples_md5 = DataManager.md5(self.samples_fn)
        samples_fn_hdf = os.path.join( os.path.dirname(self.samples_fn), 'Samples_%s.hd5'%samples_md5)
        if os.path.isfile(samples_fn_hdf):
            print('Reading samples from %s' % samples_fn_hdf)
            store = pd.HDFStore(samples_fn_hdf)
            samples = store['samples']
            store.close()
        else:
            print('Reading samples from %s' % self.samples_fn)
            xlsx = pd.ExcelFile(self.samples_fn)
            samples = pd.read_excel(xlsx, 'Samples').set_index('Sample')
            samples.sort_index(inplace=True)

            store = pd.HDFStore(samples_fn_hdf)
            store['samples'] = samples
            store.close()

        results_md5 = DataManager.md5(self.results_fn)
        results_fn_hdf = os.path.join( os.path.dirname(self.results_fn), 'Results_%s.hd5'%results_md5)
        if os.path.isfile(results_fn_hdf):
            print('Reading results from %s' % results_fn_hdf)
            store = pd.HDFStore(results_fn_hdf)
            results = store['results']
            store.close()
        else:
            print('Reading results from %s' % self.results_fn)
            xlsx = pd.ExcelFile(self.results_fn)
            results = pd.read_excel(xlsx, sheetname).set_index('Sample')
            results.index = pd.Series(results.index).fillna(method='ffill') # Account for merged cells
            results.sort_index(inplace=True)

            store = pd.HDFStore(results_fn_hdf)
            store['results'] = results
            store.close()

        self.data = pd.merge(samples, results, left_index=True, right_index=True).reset_index()

        # Remove zeros
        if self.remove_zeros:
            self.data_prime = self.data.loc[self.data[self.data.Ycol] >0, ]
        else:
            self.data_prime = self.data

        self.names = results.columns.values


        # Leave some data out for cross validation
        nSamp = len( self.data['Sample'].unique() )
        nTest = nSamp - int(round(self.training_fraction * nSamp))

        self.data['DataMode'] = 'Train'
        self.data.set_index('Sample', inplace=True)
        if len(self.data.loc[0].shape) == 1:
            nRep = 1
        else:
            nRep = self.data.loc[0].shape[0]
        self.data.loc[nSamp-nTest:, 'DataMode'] = 'Test'
        self.data.reset_index(inplace=True)
        self.data.set_index('DataMode', inplace=True)

        if self.data.shape[0] != nSamp * nRep:
            print("Shape mismatch!  Data is %d long, but nSamp=%d x nRep=%d = %d\n" % (self.data.shape[0], nSamp, nRep, nSamp*nRep))
            cnt = self.data.groupby('Sample')['Sim_Id'].count()
            print('Samples with too few reps follow:\n', cnt[cnt < nRep].sort_values())
            raise Exception('Dimension mismatch')

        print("Sim contains %d unique parameter configurations, each of which is repeated %d times." % (nSamp, nRep))

        #print(self.data.head())
        #self.X = self.Xf[:-nTest,]  # Train (samples 0 to nTest-1)
        #self.Y = self.Yf[:-nTest,]

        #self.Xt = self.Xf[-nTest:,]  # Test (the last nTest samples)
        #self.Yt = self.Yf[-nTest:,]

        print("--> Training with %d unique parameter configurations (%d simulations including replicates)"  % (nSamp-nTest, (nSamp-nTest)*nRep))
        print("--> Testing  with %d unique parameter configurations (%d simulations including replicates)" % (nTest, nTest*nRep))

        #print(self.data.loc['Train', self.data.Xcols].head())

        #store = pd.HDFStore('design_info.h5')
        #print(store)
        #print('DI\n', store.get('design_info'))
        #fitted_model.model.data.orig_exog.design_info = store['design_info']
        #store.close()

        #(self.N, self.D) = self.X.shape
        #self.D = len(self.data.Xcols)


    def transform(self, varname, func, prefix):
        new_varname = prefix + '_' + varname
        self.data[new_varname] = func( self.data[varname] )

    def rename(self, oldname, newname):
        self.data.rename(columns={oldname: newname}, inplace=True)

    def get_training_data(self):
        return self.data.copy().loc['Train', :]

    def get_test_data(self):
        return self.data.copy().loc['Test', :]

