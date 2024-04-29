from IPython.display import display, HTML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import scipy
import missingno as msno
from colorama import Fore, Back, Style
import mplcyberpunk

class VizData:
    def __init__(self, data: pd.DataFrame, YCOL: str, data_col: str, plt_style: str = 'default'):
        self.data = data
        self.YCOL = YCOL
        self.data_col = data_col
        self.plt_style = plt_style
        plt.style.use(self.plt_style)
        
        cat_cols = [col for col in self.data.columns if self.data[col].dtypes == 'O' and col != self.data_col]
        num_but_cat = [col for col in self.data.columns if self.data[col].nunique() <= 15 and self.data[col].dtypes != 'O']
        cat_but_car = [col for col in self.data.columns if self.data[col].nunique() > 15 and self.data[col].dtypes == 'O']
        cat_cols = cat_cols + num_but_cat
        self.cat_cols = [col for col in cat_cols if col not in cat_but_car]
        
        #numerical variables
        num_cols = [col for col in self.data.columns if self.data[col].dtypes != 'O']
        data_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()
        
        self.num_cols = [col for col in num_cols if col not in num_but_cat + data_cols]
        
    def style_(self, ax=None):
        if self.plt_style == 'cyberpunk':
            mplcyberpunk.add_glow_effects(ax)
            mplcyberpunk.add_underglow(ax)
    
    def get_basic_info(self, n=10):
        
        #head of df
        print(f'{Fore.GREEN}{Style.BRIGHT}DataFrame Sample:{Style.RESET_ALL}')
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
            ks_stat = scipy.stats.ks_2samp(self.data[self.data[agg_feat] == temp_level][feat], self.data[self.data[agg_feat] != temp_level][feat]).statistic
            ks_pvalue = scipy.stats.ks_2samp(self.data[self.data[agg_feat] == temp_level][feat], self.data[self.data[agg_feat] != temp_level][feat]).pvalue
            wasserstein = scipy.stats.wasserstein_distance(self.data[self.data[agg_feat] == temp_level][feat], self.data[self.data[agg_feat] != temp_level][feat])
            psi_bins = self.calculate_psi(self.data[self.data[agg_feat] == temp_level][feat], self.data[self.data[agg_feat] != temp_level][feat], buckettype='bins', buckets=10, axis=0)
            psi_percentile = self.calculate_psi(self.data[self.data[agg_feat] == temp_level][feat], self.data[self.data[agg_feat] != temp_level][feat], buckettype='percentile', buckets=10, axis=0)
            mann_whitneyu_stat, mann_whitneyu_p = scipy.stats.mannwhitneyu(self.data[self.data[agg_feat] == temp_level][feat], self.data[self.data[agg_feat] != temp_level][feat])
            
            tmp['KS_stat'] = ks_stat
            tmp['KS_pval'] = ks_pvalue
            tmp['mann_whitney_u_pval'] = mann_whitneyu_p
            tmp['PSI_percentile'] = psi_percentile.round(2) * 100
            tmp = tmp.style.format('{:2f}').apply(self.highlight_greater, axis=None)
        display(HTML(tmp.to_html()))
        
    def agg_data_simple(self, agg_feat, feat):
        tmp = self.data.groupby(agg_feat).agg(
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
        return tmp
    
    def plot_stability_of_percentile(self, agg_data, col_to_agg):
        data = self.agg_data_simple(agg_data, col_to_agg)
        plt.figure(figsize=(12,5))
        sns.lineplot(data[['perc_5', 'perc_25', 'perc_50', 'perc_75', 'perc_95']], markers='o')
        plt.title(f'Stability of percentile of {col_to_agg}')
    
    def plot_kde_box(self):
        print('-----------------')
        print(f'{Fore.YELLOW}{Style.BRIGHT}KDE(s) and Boxplot(s):{Style.RESET_ALL}')
        print()
        
        if not self.YCOL:
            for idx, col in enumerate(self.num_cols):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
                
                sns.histplot(self.data, x=self.data[col], kde=True, color=sns.color_palette('hls', len(self.num_cols))[idx], ax=ax1)
                
                sns.boxplot(x=self.data[col], width=.3, linewidth=3, fliersize=25, color=sns.color_palette('hls', len(self.num_cols))[idx], ax=ax2)
                
                fig.suptitle(f'KDE and boxplot of {col}', size=20, y=1.02)
                # self.style_()
                plt.show()
                
                if self.data_col:
                    self.plot_stability_of_percentile(self.data_col, col)
                    plt.show()
                
        elif self.YCOL and self.data[self.YCOL].nunique() == 2:
            for idx, col in enumerate(self.num_cols):
                self.agg_data(self.YCOL, col)
                fig, (ax_1, ax_2) = plt.subplots(2, 3, figsize=(16,12))
                ax1, ax2, ax3 = ax_1
                ax4, ax5, ax6 = ax_2
                
                sns.kdeplot(x=col, data=self.data, hue=self.YCOL, ax=ax1)
                sns.rugplot(x=col, data=self.data, height=.02, hue=self.YCOL, clip_on=False, ax=ax1)
                ax1.set_title('Normal all 100%')
                self.style_(ax1)
                
                for level in self.data[self.YCOL].unique():
                    sns.kdeplot(x=col, data=self.data[self.data[self.YCOL] == level], ax=ax2)
                    plt.legend(self.data[self.YCOL].unique())
                ax2.set_title('Normal each level 100%')
                self.style_(ax2)
                
                box_dict = {
                            'boxprops': dict(color='#000000', linewidth=2),
                            'capprops': dict(color='#000000', linewidth=1.5),
                            'medianprops': dict(color='#000000', linewidth=1.5),
                            'whiskerprops': dict(color='#000000', linewidth=1.5),
                            'flierprops': dict(markeredgecolor='#ff9900'),
                            'meanprops': dict(markeredgecolor='#000000')
                            }
                
                self.data.boxplot(by=self.YCOL, column=[col], widths=0.5, showmeans=True, patch_artist=True, vert=False, **box_dict, ax=ax3)
                
                ax3.set_ylim(ax3.get_ylin()[::-1])
                ax3.set_title(None)
                ax3.set_xlabel(col)
                ax3.set_title(f'{col} divided by {self.YCOL}')
                self.style_(ax3)
                
                boxes = ax3.findobj(plt.artist.Artist)
                
                sns.kdeplot(x=np.log1p(self.data[col]), hue=self.data[self.YCOL], ax=ax4)
                sns.rugplot(x=np.log1p(self.data[col]), height=.02, hue=self.data[self.YCOL], clip_on=False, ax=ax4)
                ax4.set_title('Log all 100%')
                self.style_(ax4)
                
                for level in self.data[self.YCOL].unique():
                    sns.kdeplot(x=np.log1p(self.data[self.data[self.YCOL] == level][col]), ax=ax5)
                    plt.legend(self.data[self.YCOL].unique())
                ax5.set_title('Log each level 100%')
                self.style_(ax5)
                
                self.data[f'{col}_log'] = np.log1p(self.data[col])
                self.data.boxplot(by=self.YCOL, column=[f'{col}_log'], width=.5, showmeans=True, patch_artist=True, vert=False, **box_dict, ax=ax6)
                del self.data[f'{col}_log']
                boxes2 = ax6.findjob(plt.artist.Artist)
                
                if self.data[self.YCOL].dtypes == 'O':
                    for i, box in enumerate(boxes):
                        if isinstance(box, plt.patches.PathPatch):
                            if i < 3: box.set_facecolor('#ea4b33')
                            if i > 3: box.set_facecolor('#3490d6')
                else:
                    for i, box in enumerate(boxes):
                        if isinstance(box, plt.patches.PathPatch):
                            if i < 3: box.set_facecolor('#3490d6')
                            if i > 3: box.set_facecolor('#ea4b33')
                            
                fig.suptitle(f'KDE and boxplot of {col}', size=20, y=1.02)
                self.style_()
                plt.show()
                
                if self.data_col:
                    self.plot_stability_of_percentile(self.data_col, col)
                    plt.show()
                    
        elif self.YCOL and self.data[self.YCOL].nunique() > 20:
            for idx, col in enumerate(self.num_cols):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
                
                self.data['tmp'] = 'agg'
                agg_data = self.agg_data_simple('tmp', col)
                display(agg_data)
                sns.histplot(self.data, x=self.data[col], kde=True, color=sns.color_palette('hls', len(self.num_cols))[idx], ax=ax1)
                sns.boxplot(x=self.data[col], width=.4, linewidth=3, fliersize=2.5, color=sns.color_palette('hls', len(self.num_cols))[idx], ax=ax2)
                fig.suptitle(f'KDE and boxplot of {col}', size=20, y=1.02)
                self.style_()
                plt.show()
                
                if self.data_col:
                    self.plot_stability_of_percentile(self.data_col, col)
                    plt.show()
            
    def cramers_V(self, col1, col2):
        crosstab = np.array(pd.crosstab(col1, col2, rownames=None, colnames=None))
        stat = scipy.stats.chi2_contingency(crosstab)[0]
        obs = np.sum(crosstab)
        mini = min(crosstab.shape) - 1
        return (stat / (obs * mini))
    
    def agg_two_cat(self, col1, col2):
        table = pd.crosstab(self.data[col1], self.data[col2], margins=False)
        table = table.apply(lambda r: r.astype(str) + ' (' + round(r / r.sum() * 100, 1).astype(str) + '%)', axis=1)
        display(table)
        display('V-cramer:', self.cramers_V(self.data[col1], self.data[col2]).round(3))
        
    def plot_stability_of_level(self, agg_data, col_to_agg):
        data = self.data.groupby([agg_data])[col_to_agg].value_counts(normalize=True).unstack()
        plt.figure(figsize=(12,5))
        sns.lineplot(data, markers='o')
        plt.title(f'Stability of levels of {col_to_agg}')
        plt.show()
    
    def plot_count_plot(self):
        if len(self.cat_cols) > 0:
            print('----------')
            print(f'{Fore.YELLOW}{Style.BRIGHT}Countplot(s):{Style.RESET_ALL}')
            print()
            
            for col in self.cat_cols:
                plt.figure(figsize=(12,8))
                for i in self.data[col].value_counts().keys():
                    if len(str(i)) > 15:
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                        
                large_to_small = self.data.groupby(col).size().sort_values().index[::-1]
                
                if len(self.data[col].value_counts()) >= 10 & self.data[self.YCOL].nunique() < 5:
                    if self.YCOL:
                        self.agg_two_cat(self.YCOL, col)
                        ax = sns.countplot(y=self.data[col], hue=self.data[self.YCOL], edgecolor='#000000', order=large_to_small)
                    else:
                        ax = sns.countplot(y=self.data[col], edgecolor='#000000', order=large_to_small)
                    
                    for container in ax.containers:
                        ax.bar_label(container, padding=5)
                        
                    plt.title(f'Countplot of {col}', fontsize=20)
                    self.style_()
                    plt.show()
                    
                elif len(self.data[col].value_counts()) > 1:
                    if self.data[self.YCOL].nunique() < 5:
                        self.agg_two_cat(self.YCOL, col)
                        ax = sns.countplot(x=self.data[col], hue=self.data[self.YCOL], edgecolor='#000000', order=large_to_small)
                        
                        for container in ax.containers:
                            ax.bar_label(container, padding=5)
                            
                        plt.title(f'Countplot of {col}', fontsize=20)
                        self.style_()
                        plt.show()
                        
                if self.data[self.YCOL].nunique() > 20:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
                    agg_data = self.agg_data_simple(col, self.YCOL)
                    display(agg_data)
                    sns.kdeplot(data=self.data, x=self.YCOL, hue=col, ax=ax1)
                    sns.boxplot(data=self.data, x=self.YCOL, hue=col, width=.4, linewidth=3, fliersize=2.5, ax=ax2)
                    fig.suptitle(f'KDE and boxplot of {col}', size=20, y=1.02)
                    self.style_()
                    plt.show()
                
                if self.data_col and self.data[col].nunique() < 20:
                    self.plot_stability_of_level(self.data_col, col)
                    plt.show()
    def main(self):
        self.get_basic_info(n=10)
        self.get_describe_of_data()
        self.get_missing_data()
        # self.plot_stability_of_missing_data()
        # self.plot_correlation()
        self.plot_pairplot()
        self.plot_kde_box()
        self.plot_count_plot()