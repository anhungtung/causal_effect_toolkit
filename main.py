import os 

import numpy as np
import pandas as pd 

from methods.base_causal import baseTreatmentEffect

if __name__ == '__main__':
    X = pd.read_csv('data.csv')
    X['version'] = [1 if x == 'gate_40' else 0 for x in X['version']]
    X['retention_7'] = [1 if x == True else 0 for x in X['retention_7']]
    X['retention_1'] = [1 if x == True else 0 for x in X['retention_1']]
    print(X)

    bTE = baseTreatmentEffect(data=X, unit_col='userid', treatment_col='version', outcome_col='retention_7')
    bTE.empirical_results()