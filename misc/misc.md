CutNearSamples potential testing code:
```python3
        #TODO(dklein): What is the purpose of the code below? Can it be removed?
        print 'Drawing %d samples:'%nSamples

        seeds = self.seeds.sample(n=nSamples, replace=True).reset_index(drop=True)
        sample = seeds.copy()

        for row_idx, row in seeds.iterrows():
            for (col_name, value) in row.iteritems():
                v = self.param_info.loc[col_name]
                v_range = (v['Max']-v['Min'])

                right_frac = (v['Max'] - value) / v_range
                left_frac = (value - v['Min']) / v_range

                blur_frac = np.min([self.blur_fraction_of_range, right_frac, left_frac])

                sample.loc[row_idx, col_name] = value + \
                        np.random.uniform(
                            low = -blur_frac * v_range,
                            high = blur_frac * v_range,
                            size = 1 )

        print 'DONE drawing %d samples:'%nSamples

        return sample

        print 'in draw_samples, nSamples=%d' % nSamples

        good = np.ones(nSamples, dtype=bool)
        sample = self.seeds.sample(n=nSamples, replace=True).reset_index(drop=True)
        print 'in draw_samples, sample len is %d' % sample.shape[0]
        print 'in draw_samples, here is sample head:\n', sample.head()

        for i, xc in enumerate(self.Xcols_all_orig):
            v = self.param_info.loc[xc]
            print 'in draw_samples, here is v:\n', v
            print 'blur fraction of range is %f', self.blur_fraction_of_range
            sample[xc] += \
                np.random.uniform(
                    low=-self.blur_fraction_of_range*(v['Max']-v['Min']),
                    high=self.blur_fraction_of_range*(v['Max']-v['Min']),
                    size=sample.shape[0] )

            # Resample points that are outside of Min-Max
            df = (sample[xc] > v['Min']) & (sample[xc] < v['Max'])
            good &= df.values

        print 'in draw_samples, good len is %d, sum is %d' % (good.shape[0], sum(good))
        return sample.loc[good]
```