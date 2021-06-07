import os 

import numpy as np
import pandas as pd 
from scipy.special import comb

class baseTreatmentEffect:
    def __init__(self, data, unit_col, treatment_col, outcome_col, time_col=None):
        self.data = data

        self.unit_col = unit_col
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col 
        self.time_col = time_col

        self.n_user = len(set(self.data[self.unit_col]))
        pass 
    
    def calculate_SATE(self):
        outcome_with_treatment = sum(self.data[self.data[self.treatment_col] == 1][self.outcome_col])
        outcome_without_treatment = sum(self.data[self.data[self.treatment_col] == 0][self.outcome_col])

        return 1/self.n_user*(outcome_with_treatment-outcome_without_treatment)
    
    def _variance_DiM(self, res):
        wo_avg = np.average(self.data[self.data[self.treatment_col] == 1][self.outcome_col])
        w_avg = np.average(self.data[self.data[self.treatment_col] == 0][self.outcome_col])

        wt = self.data[self.data[self.treatment_col] == 1][self.outcome_col] - w_avg
        wot = self.data[self.data[self.treatment_col] == 0][self.outcome_col]- wo_avg

        s_0 = 1/(self.n_user-1)*sum(wot*wot)
        s_1 = 1/(self.n_user-1)*sum(wt*wt)
        
        try:
            s_01 = np.cov(wt, wot)
        except:
            return 'Not Identifiable' 

        return 1/res['N']*(res['N0']/res['N1']*s_1+res['N1']/res['N0']*s_0+2*s_01)
    
    def calculate_SATT(self):
        return
    
    def _binary_table(self):
        """ Only Binary Outcome """ 
        bt = self.data.groupby(by=[self.treatment_col, self.outcome_col])[self.unit_col].count().reset_index()
        bt.columns = ['Treatment', 'Outcome', 'Count_Units']

        res = {
            'table':bt, 
            'N': sum(bt['Count_Units']),
            'N1': sum(bt[bt['Treatment'] == 1]['Count_Units']),
            'N0' : sum(bt['Count_Units']) -  sum(bt[bt['Treatment'] == 1]['Count_Units']),
            'TestStat':sum(bt[(bt['Treatment'] == 1) & (bt['Outcome'] == 1)]['Count_Units']),
            'M' : sum(bt[bt['Outcome'] == 1]['Count_Units']),
            'turnout_control' : sum(bt[(bt['Treatment'] == 0) & (bt['Outcome'] == 1)]['Count_Units'])/sum(bt[(bt['Treatment'] == 0)]['Count_Units']),
            'turnout_test' : sum(bt[(bt['Treatment'] == 1) & (bt['Outcome'] == 1)]['Count_Units'])/sum(bt[(bt['Treatment'] == 1)]['Count_Units'])
        }

        return res

    def empirical_results(self, verbose=True, mode='binary'):
        res = self._binary_table()

        esti = res['turnout_test'] - res['turnout_control']
        lift = res['turnout_test'] / res['turnout_control'] - 1
        se = np.sqrt(res['turnout_test'] * (1-res['turnout_test']) / res['N1'] + res['turnout_control'] * (1-res['turnout_control']) / res['N0'])

        out = {
            'Control Baseline' :  "{:.4%}%".format(float(res['turnout_control'])),
            'Treated Outcome' :  "{:.4%}%".format(float(res['turnout_test'])),
            'Point Effect' : "{:.4%}%".format(float(esti)),
            'Lift Effect' : "{:.4%}%".format(float(lift)),
            'Standard Deviation' : se,
            '95% C.I.' : ["{:.4%}%".format(float(esti - 1.96*se)), "{0:.0f}%".format(float(esti + 1.96*se))]
        }

        if verbose:
            print('======= Empirical Test Effect =======')
            for key, item in out.items():
                print(key + ' : %s ' % (item))

            print('Var Estimator : ', self._variance_DiM(res))
            
            print('=====================================')
        
        return out