from IPython.display import display, HTML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import scipy
import missingno as msno
from colorama import Fore, Back, Style

class VizData:
    def __init__(self, data: pd.DataFrame, YCOL: str, data_col: str, plt_style: str = 'default'):
        self.data = data
        self.YCOL = YCOL
        self.data_col = data_col
        self.plt_style = plt_style
        plt.style.use(self.plt_style)
        
        categorical_columns = [col for col in self.data.columns if self.data[col].dtypes == 'O' and col != self.data_col]
        
    def get_basic_info(self, n=10):
        
        #head of df
        print(f'{Fore.GREEN}{Style.Bright}DataFrame Sample:{Style.RESET_ALL}')
        display(self.data.sample(n))
        
        #shape of df
        print(f'{Fore.GREEN}{Style.BRIGHT}Shape of DataFrame:{Style.RESET_ALL}')
        print()
        print(self.data.shape)
        print('-------------------')
        
        #info about df
        print(f'{Fore.GREEN}{Style.BRIGHT}DataFrame info:{Style.RESET_ALL}')
        print()
        print(self.data.info())
        print('-------------------')

    def get_describe_of_data(self):
        print(f'{Fore.BLUE}{Style.BRIGHT}Describe of DataFrame:{Style.RESET_ALL}')
        print()
        
        def color_negative_red(value):
            if value < 0:
                color = '#ff0000'
            elif value > 0:
                color = '#00ff00'
            else:
                color = '#FFFFFF'
            return 'color: %s' % color
    
        skewness = self.data._get_numeric_data().dropna().apply(lambda x: scipy.stats.skew(x)).to_frame(name='skewness')
        kurtosis = self.data._get_numeric_data().dropna().apply(lambda x: scipy.stats.kurtosis(x)).to_frame(name='kurtosis')
        
        skewness_and_kurtosis_df = pd.concat([skewness, kurtosis], axis=1)
        
        describe_df = self.data.describe().T
        
        full_info = pd.concat([describe_df, skewness_and_kurtosis_df], ignore_index=True, axis=1)
        full_info.columns = list(describe_df.columns) + list(skewness_and_kurtosis_df.columns)
        full_info.insert(loc=2, column='median', value=self.data.median(skipna=True, numeric_only=True))
        
        full_info.iloc[:, :-2] = full_info.iloc[:, :-2].applymap(
            lambda x: format(x, '.3f').rstrip('0').rstrip('.') if isinstance(x, (int, float)) else x
        )
        
        info_cols = ['skewness', 'kurtosis']
        
        display(full_info.style.background_gradient(cmap='Spectral', subset=full_info.columns[:-2])
                .applymap(color_negative_red, subset=info_cols)
                .set_properties(**{'bakground-color': '#000000', 'font-weight': 'bold'}, subset=info_cols)
                .set_properties(**{'font-weight': 'bold'}, subset=full_info.columns[:-2]))
        print('-------------------')
        
    def plot_stability_of_missing_data(self):
        columns_with_empties = [col for col in self.data.columns if self.data[col].isna().sum() > 0]
        
        if self.data_col:
            for feature in columns_with_empties:
                n_missing = self.data[self.data[feature].isnull()].groupby([self.data_col]).size().rename('n_missing').reset_index(drop=True)
                n_obs = self.data.groupby([self.data_col]).size().rename('n_obs').reset_index(drop=True)
                n_missing = n_missing.merge(n_obs, on=self.data_col)
                n_missing['n_missing_percentage'] = n_missing['n_missing'] / n_missing['n_obs']
                
                sns.lineplot(x=self.data_col, y='n_missing_percentage', data=n_missing, marker='.')
                plt.title(f'Stability of missing feature: {feature}')
                plt.xticks(rotation=70)
                plt.axhline(y=.2, color='r', linestyle='-')
                plt.axhline(y=.5, color='r', linestyle='-')
                plt.axhline(y=.8, color='r', linestyle='-')
                plt.ylim([0,1])
                plt.show()
                
        # to finish