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
                
    def get_missing_data(self):
        print(f'{Fore.RED}{Style.BRIGHT}Missing values:{Style.RESET_ALL}')
        
        columns_with_empties = [col for col in self.data.columns if self.data[col].isna().sum() > 0]
        df0 = self.data[columns_with_empties]
        df0 = df0.isna().sum().to_frame()
        df0.columns= ['n_missing']
        df0['n_missing_percentage'] = round(df0['n_missing'] / self.data.shape[0] * 100, 2).astype(str)
        df0['n_missing_percentage'] = df0['n_missing_percentage'] + '%'
        
        display(df0.style.set_properties(**{'background_color': '#000000', 'color': '#ff0000', 'font_weight': 'bold'}))
        print('-------------------')
        
        #MSNO matrix
        if self.data.shape[1] < 100:
            if len(columns_with_empties) > 0:
                print(f'{Fore.RED}{Style.BRIGHT}MSNO Matrix:{Style.RESET_ALL}')
                print()
                ax = msno.matrix(self.data, color=(0, 0.5, 0), figsize=(12,8))
                plt.show()
                print('-------------------')
    
    def plot_correlation(self):
        print(f'{Fore.BLUE}{Style.BRIGHT}DataFrame Correlation:{Style.RESET_ALL}')
        print()
        
        if self.data.columns.nunique() < 10:
            sns.heatmap(self.data.corr(), annot=True, cmap='Spectral', linewidths=2, linecolor='#000000', fmt='.3f')
            plt.show()
        elif self.data.columns.nunique() < 15:
            sns.heatmap(self.data.corr(), annot=True, cmap='Spectral', linewidths=2, linecolor='#000000', fmt='.2f')
            plt.show()
        elif self.data.columns.nunique() < 25:
            sns.heatmap(self.data.corr(), annot=True, cmap='Spectral', linewidths=2, linecolor='#000000', fmt='.1f')
            plt.show()
        else:
            sns.heatmap(self.data.corr(), annot=True, cmap='Spectral')
            plt.show()
            
    def plot_pairplot(self):
        if self.data.columns.nunique() < 15:
            print('-------------------')
            print(f'{Fore.BLUE}{Style.BRIGHT}DataFrame Pairplot:{Style.RESET_ALL}')
            print()
            
            not_bool_columns = [col for col in self.data.columns if self.data[col].dtypes != bool]
            df_bool = self.data[not_bool_columns]
            
            if self.YCOL:
                sns.pairplot(df_bool, hue=self.YCOL, palette=sns.color_palette('hls', self.data[self.YCOL].nunique()))
                plt.show()
            else:
                sns.pairplot(df_bool)
                plt.show()
                
    def highlight_greater(self, x):
        m1 = x['KS_stat'] > 0.5
        m2 = x['ks_pvalue'] < 0.05
        m3 = x['PSI_bins'] > 25
        m4 = x['PSI_percentile'] > 25
        
        df1 = pd.DataFrame(index=x.index, columns=x.columns)
        
        df1['KS_stat'] = np.where(m1, 'background-color {}'.format('lightgreen'), 'background-color {}'.format('salmon'))
        df1['ks_pvalue'] = np.where(m2, 'background-color {}'.format('lightgreen'), 'background-color {}'.format('salmon'))
        df1['mannwhitneyu_p'] = np.where(m2, 'background-color {}'.format('lightgreen'), 'background-color {}'.format('salmon'))
        df1['PSI_bins'] = np.where(m3, 'background-color {}'.format('lightgreen'), 'background-color {}'.format('salmon'))
        df1['PSI_percentile'] = np.where(m4, 'background-color {}'.format('lightgreen'), 'background-color {}'.format('salmon'))
        
        return df1

    def calculate_psi(self, expected, actual, buckettype='bins', buckets=10, axis=0):
        """_summary_

        Args:
            expected (_type_): _description_
            actual (_type_): _description_
            buckettype (str, optional): _description_. Defaults to 'bins'.
            buckets (int, optional): _description_. Defaults to 10.
            axis (int, optional): _description_. Defaults to 0.
        """
        
        def psi(expected_array, actual_array, buckets):
            """_summary_

            Args:
                expected_array (_type_): _description_
                actual_array (_type_): _description_
                buckets (_type_): _description_
            """
            
            def scale_range(input, min, max):
                input += -(np.min(input))
                input /= np.max(input) / (max - min)
                input += min
                return input
            
            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
            
            if buckettype == 'bins':
                breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
                
            elif buckettype == 'quantiles':
                breakpoints = np.stack(
                    [np.percentile(expected_array, b) for b in breakpoints]
                )
                
            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
            
            def sub_psi(e_perc, a_perc):
                if a_perc == 0:
                    a_perc = 0.0001
                if e_perc == 0:
                    e_perc = 0.0001
                
                value = (e_perc - a_perc) * np.log(e_perc / a_perc)
                return value
        
            psi_value = np.sum(
                sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))
            )
            
            return (psi_value)
        
        if len(expected.shape) == 1:
            psi_values = np.empty(len(expected.shape))
        else:
            psi_values = np.empty(expected.shape[axis])
            
        for i in range(0, len(psi_values)):
            if len(psi_values) == 1:
                psi_values = psi(expected, actual, buckets)
            elif axis == 0:
                psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
            elif axis == 1:
                psi_values[i] = psi(expected[i, :], actual[i, :], buckets)
        
        return (psi_values)
    
    def agg_data(self, agg_feat, feat):
        # to optimize with a loop
        tmp = self.data.groupby([agg_feat]).agg(
            min_ = (feat, min),
            perc_5 = (feat, lambda x: np.percentile(x, 5)),
            perc_25 = (feat, lambda x: np.percentile(x, 25)),
            perc_50 = (feat, lambda x: np.percentile(x, 50)),
            mean = (feat, 'mean'),
            perc_75 = (feat, lambda x: np.percentile(x, 75)),
            perc_95 = (feat, lambda x: np.percentile(x, 95)),
            max_ = (feat, max),
            _n_ = (feat, 'count')
        )
        
        if self.data[agg_feat].nunique() == 2:
            temp_level = self.data[agg_feat].unique()[0]
            ks_stat = scipy.stats.ks_2samp(self.data[self.data[agg_feat] == temp_level])