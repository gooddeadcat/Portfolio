import pathlib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import pickle


DATA_DIR = pathlib.Path(".")


p_0_0 = pickle.load(open(pathlib.Path(__file__).parent.joinpath("p_0_0.pickle"), "rb"))
p_0_1 = pickle.load(open(pathlib.Path(__file__).parent.joinpath("p_0_1.pickle"), "rb"))
p_1_0 = pickle.load(open(pathlib.Path(__file__).parent.joinpath("p_1_0.pickle"), "rb"))
p_1_1 = pickle.load(open(pathlib.Path(__file__).parent.joinpath("p_1_1.pickle"), "rb"))


def predict(df: pd.DataFrame) -> pd.DataFrame:   
    df.feature4.replace({'gas1': 0, 'gas2': 1}, inplace=True)
    
    df0 = df.query('feature4 == 0').copy()
    df1 = df.query('feature4 == 1').copy()
    
    
    MIX = ['feature13', 
           'feature22',
          ]
    
    predictions0 = {}
    predictions0['target0'] = p_0_0.predict(df0) * df0[MIX].apply(lambda x: sum(x), axis=1)
    predictions0['target1'] = p_0_1.predict(df0) * df0[MIX].apply(lambda x: sum(x), axis=1)
    df0['target0'] = predictions0['target0']
    df0['target1'] = predictions0['target1']
    
    predictions1 = {}
    predictions1['target0'] = p_1_0.predict(df1) * df1[MIX].apply(lambda x: sum(x), axis=1)
    predictions1['target1'] = p_1_1.predict(df1) * df1[MIX].apply(lambda x: sum(x), axis=1)
    df1['target0'] = predictions1['target0']
    df1['target1'] = predictions1['target1']
    
    df2 = pd.concat([df0, df1], axis=0)
    df2.sort_index(inplace=True)
    
    preds_df = df2[['target0', 'target1']]
    
    return preds_df