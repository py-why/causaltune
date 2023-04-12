import math
from typing import Optional, Dict, Union, Any, List, Tuple, Literal

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from auto_causality.shap import shap
from auto_causality.models.dummy import PropensityScoreWeighter


# TODO: decide on how Visualizer and AutoCausality should interact; 
#       currently standalone and mimicking some methods from auto-causality
# TODO: make plots themselves a bit nicer and allow for more flexibility/customization
# TODO: (?) write tests

class Visualizer:

    def __init__(
        self,
        test_df: pd.DataFrame,
        treatment_col_name: str,
        outcome_col_name: str,
                 
    ) -> None:
        
        assert treatment_col_name and outcome_col_name in test_df.columns, 'Check that treatment and outcome columns are specified correctly.'
        
        self.test_df = test_df
        self.treatment_col_name = treatment_col_name
        self.outcome_col_name = outcome_col_name
        self.t_list = [int(t) for t in sorted(list(self.test_df[self.treatment_col_name].unique()))]


    # TODO: make this a bit nicer
    def plot_shap(
        self,
        shapley_values: np.ndarray,
        df: pd.DataFrame,
        effect_modifier_names: List,
        figtitle: str = ""         
    
    ) -> None:
        
        assert len(effect_modifier_names) == shapley_values.shape[1], "Column mismatch."
        
        plt.title(figtitle)
        shap.summary_plot(shapley_values, df[effect_modifier_names])
        plt.show()


    def plot_metrics_by_estimator(
        self,
        scores_dict: Dict,
        figtitle: str = "",
        metrics: Tuple[str, str] = ('norm_erupt', 'ate'),
        figsize: Tuple[int, int] = (7,5),
        
    ) -> None:
        
        """
        Plot metrics by estimator.

        @param scores_dict: scores dict from fitted auto-causality object;
        @param figtitle: str to specify plot title, defaults to "";
        @param metrics: tuple specifying metrics, defaults to ('norm_erupt', 'ate');
        @param figsize: tuple specifying plot size, defaults to (7,5);
        
        """
        
        colors = ([matplotlib.colors.CSS4_COLORS['black']] +
            list(matplotlib.colors.TABLEAU_COLORS) + [
            matplotlib.colors.CSS4_COLORS['lime'],
            matplotlib.colors.CSS4_COLORS['yellow'],
            matplotlib.colors.CSS4_COLORS['pink']
        ])


        plt.figure(figsize = figsize)
        plt.title(figtitle)

        m1 = metrics[1]
        m2 = metrics[0]

        for (est, scr), col in zip(scores_dict.items(),colors):
            try:
                sc = [scr["scores"]['train'][m1], scr["scores"]['validation'][m1], scr["scores"]['test'][m1]]
                crv = [scr["scores"]['train'][m2], scr["scores"]['validation'][m2], scr["scores"]['test'][m2]]
                plt.plot(sc, crv, color=col, marker="o", label=est)
                plt.scatter(sc[1:2],crv[1:2], c=col, s=70, label="_nolegend_" )
                plt.scatter(sc[2:],crv[2:], c=col, s=120, label="_nolegend_" )

            except:
                pass
        plt.xlabel(m1)
        plt.ylabel(m2)

        plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        plt.annotate('By size (ascending): train/val/test', (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')

        plt.grid()
        plt.show()
        

    def plot_group_ate(
        self,
        policy_df: pd.DataFrame,
        psw: PropensityScoreWeighter,
        single_t: str = ''

    ) -> Dict:
        
        """
        Plot group ate per treatment.

        @param policy_df: pd.DataFrame with cols T, Y, policy_T_1, policy_T_2 etc., policy cols being bool;
        @param psw: PropensityScoreWeighter from auto-causality;
        @param single_t: if set to specific treatment, only plot group ate for this treatment; defaults to '' to generate joint plot;
        
        @return all_grps: Dict containing treatments as keys and group ate dataframes as values;
        
        """
        
        assert self.treatment_col_name and self.outcome_col_name in policy_df.columns, f"Please add columns {self.treatment_col_name} and {self.outcome_col_name} to your policy dataframe."
        assert self.test_df[self.treatment_col_name].nunique()-1 == len(policy_df.columns)-2, f'Mismatch between number of treatments ({self.test_df[self.treatment_col_name].nunique()-1}) and supplied policies ({len(policy_df.columns)-2}).'

        # obtain group ate df for every treatment
        all_grps = {}
        for t in policy_df.columns:
            if t not in [self.treatment_col_name, self.outcome_col_name]:
                t_pol = policy_df[(policy_df[self.treatment_col_name] == int(t)) | (policy_df[self.treatment_col_name] == 0)][[t]]
                t_pol.reset_index(inplace=True)
                all_grps[t] = self.calculate_group_ate(t, t_pol, psw)


        # re-shuffle initial dict for plotting
        tmp = pd.DataFrame(columns=['policy', 'mean', 'std', 'count'])
        for v in all_grps.values():
            tmp = pd.concat([tmp, v],ignore_index=True)

        d = {}
        for p in tmp['policy'].unique():
            d[p] = {'mean': [], 'std': []}
            _ = tmp[tmp['policy']==p]
            for m in ['mean', 'std']:
                d[p][m] = _[m].values
        
        # actual plotting
        if single_t == '':
            self.__plot_multi_ate(grp_dict=all_grps, plot_dict=d)
        else:
            self.__plot_single_ate(grp_dict=all_grps, single_t=single_t)


        return all_grps



    def __plot_multi_ate(
        self,
        grp_dict: Dict,
        plot_dict: Dict
    ):
        X = [str(t) for t in grp_dict.keys()]
        plt.figure(figsize=(3*len(X),6))

        if not False:
            m, c, b = plt.errorbar(X, plot_dict['False']['mean'], yerr=plot_dict['False']['std'], color='blue', fmt='o', elinewidth=3)
            [bar.set_alpha(0.5) for bar in b]
            m, c, b = plt.errorbar(X, plot_dict['True']['mean'], yerr=plot_dict['True']['std'], color='red', fmt='o', elinewidth=3 )
            [bar.set_alpha(0.5) for bar in b]

            plt.legend(['True', 'False'], loc='best')

        # TODO: make this actually work
        else:
            m, c, b = plt.errorbar(X, plot_dict['all']['mean'], yerr=plot_dict['all']['std'], color='black', fmt='o', elinewidth=3 )
            [bar.set_alpha(0.5) for bar in b]

            plt.legend(['all'], loc='best')

        plt.grid()
        # plt.title('test')    
        plt.xlabel('Treatment')
        plt.ylabel('Group ATE')
        plt.show()

    def __plot_single_ate(
        self,
        grp_dict: Dict,
        single_t: str

    ):
    
        s_t = grp_dict[single_t]
        grp = s_t["policy"].unique()

        colors = (matplotlib.colors.CSS4_COLORS['black'],
                matplotlib.colors.CSS4_COLORS['red'],
                matplotlib.colors.CSS4_COLORS['blue'])

        for i,(p,c) in enumerate(zip(grp, colors)):
            st = s_t[s_t["policy"] == p]
            plt.errorbar(np.array(range(len(st))) +0.1*i, st["mean"].values[0],  yerr = st["std"].values[0], fmt='o', color=c)
        plt.legend(grp)
        plt.grid(True)
        plt.ylabel('Group ATE')
        plt.xticks([])
        plt.title(f'Treatment {single_t}')
        plt.show()
    

    # mimic ac.scorer.group_ate method
    def calculate_group_ate(
        self,
        t: int,
        t_pol: pd.DataFrame,
        psw: PropensityScoreWeighter
    ):
        
        t_df = self.test_df[(self.test_df[self.treatment_col_name] == int(t)) | (self.test_df[self.treatment_col_name] == 0)]
        t_df.reset_index(inplace=True)
        tmp = {'all': self.ate(t_df, psw)}

        for p in sorted(list(t_pol[t].unique())):
            tmp[p] = self.ate(t_df[t_pol[t] == p], psw)

        tmp2 = [
            {"policy": str(p), "mean": m, "std": s, "count": c}
            for p, (m, s, c) in tmp.items()
        ]

        return pd.DataFrame(tmp2)
 
    # mimic ac.scorer.ate method, but adjusting it slightly for multi-t
    def ate(
        self, 
        t_df: pd.DataFrame,
        psw: PropensityScoreWeighter
    ):
        
        estimate = np.nanmax(psw.effect(t_df).mean(axis=0))
        naive_est = self.naive_ate(t_df[self.treatment_col_name], t_df[self.outcome_col_name])

        return estimate, naive_est[1], naive_est[2]

    # mimic naive_ate function for sneaky std calc
    def naive_ate(
        self,
        t_col: pd.Series,
        o_col: pd.Series
    ):
        
        treated = (t_col > 0).sum()
        
        mean_ = o_col[t_col > 0].mean() - o_col[t_col == 0].mean()
        std1 = o_col[t_col > 0].std() / (math.sqrt(treated) + 1e-3)
        std2 = o_col[t_col == 0].std() / (
            math.sqrt(len(o_col) - treated) + 1e-3
        )
        std_ = math.sqrt(std1 * std1 + std2 * std2)
        return (mean_, std_, len(t_col))