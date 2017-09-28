from dtk.utils.analyzers.DownloadAnalyzer import DownloadAnalyzer
from simtools.AnalyzeManager.AnalyzeManager import AnalyzeManager
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.SetupParser import SetupParser

import json
import os
import xlsxwriter

class PrevAnalyzer(DownloadAnalyzer):
    def __init__(self, age_range, year):
        self.age_range = age_range
        self.year = int(year)

        self.filenames = [os.path.join('output', 'BinnedReport.json')]
        super(PrevAnalyzer, self).__init__(filenames=self.filenames)

    def apply(self, parser):
        """
        I'm being lazy; assuming monthly data
        :param parser:
        :return:
        """
        super(PrevAnalyzer, self).apply(parser)

        # open up the downloaded file, parse out the needed bits and return the computed prevalence
        filename = os.path.join(self.get_sim_folder(parser), os.path.basename(self.filenames[0]))
        results = json.load(open(filename, 'r'))
        meta = results['Header']

        # determine index range for selected year
        start_date = meta['Base_Year']
        timestep = 1/12.0 # in years
        start_time_index = int(round((self.year - start_date) / timestep))
        end_time_index = start_time_index + 11

        # compute average prevalance for given year
        age_index = meta['Subchannel_Metadata']['MeaningPerAxis'][0][0].index(self.age_range)
        infections = sum(results['Channels']['Infected']['Data'][age_index][start_time_index:end_time_index+1]) / 12.0
        population = sum(results['Channels']['Population']['Data'][age_index][start_time_index:end_time_index+1]) / 12.0

        prevalence = infections / population
        # print('id: %s age-range: %s infections: %s population: %s prevalence: %s' %
        #       (parser.sim_id, self.age_range, infections, population, prevalence))

        return {
            'prevalence_%s_%s'%(self.age_range, self.year): prevalence,
            'Sim_Id': parser.sim_id,
            'id': parser.sim_data['__sample_id__']
        }

    def combine(self, parsers):
        from pprint import pprint

        data = []
        for sim_id, parser in parsers.iteritems():
            this_data = parser.selected_data.values()[0]
            data.append(this_data)
        # pprint(data)
        self.combined_data = data

    def finalize(self):
        """
        Write out the Results.xlsx file expected by 'hm fit'
        :param self:
        :return:
        """
        workbook = xlsxwriter.Workbook('Results.xlsx')
        worksheet = workbook.add_worksheet('Values')

        # First, write the sheet header
        row = 0
        col = 0
        order = sorted(self.combined_data[0].keys())
        for key in order:
            worksheet.write(row, col, key)
            col += 1

        # Now write the per-simulation results, one sim per line
        row = 1
        for item in self.combined_data:
            col = 0
            for key in order:
                worksheet.write(row, col, item[key])
                col += 1
            row += 1
        workbook.close()

SetupParser.default_block = 'HPC'

analyzers = [PrevAnalyzer(age_range='40-44', year=2000)]
