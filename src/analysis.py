from collections import defaultdict
from scipy import stats
import numpy as np
import pandas as pd
# import json
import ast

from matplotlib.patches import Rectangle, Patch
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns

from llm_runner import LLMRunner, ModelDeployments, model_display_names
import utils as ut
import prompt_library as pl

llm_runner = LLMRunner()

# AGGREGATE PERMUTATIONS
def aggregate_over_permutations(perm_choice_dict, simulated_sample_size=100):
    def choice_processing(choice_dict):
        # simulated samples for gpt 3.5 (logprobs)
        sample_size = simulated_sample_size if sum(choice_dict.values())<=1.0001 else 1
        processed_dict = {k:v*sample_size for k,v in choice_dict.items()}
        
        # set default value for TARGET if not sampled
        if 'TARGET' not in processed_dict.keys():
            processed_dict['TARGET'] = 0
        return processed_dict
    
    perm_choice_dict_processed = {
        p:choice_processing(c)
        for p,c in perm_choice_dict.items()
        }
    
    # aggregate choice frequencies across permutations 
    ## average
    sum_aggregated = defaultdict(float)
    
    for d in perm_choice_dict_processed.values():
        for k,v in d.items():
            sum_aggregated[k] += v

    totals = {
        "TOTAL_NUMB_SAMPLES": sum(sum_aggregated.values()),
        "NON_TARGET": sum(sum_aggregated.values())-sum_aggregated["TARGET"],
        }
    
    aggregated_norm = {
        "TARGET_AVG_PROB": sum_aggregated['TARGET']/totals['TOTAL_NUMB_SAMPLES'],
        "NON_TARGET_AVG_PROB": totals['NON_TARGET']/totals['TOTAL_NUMB_SAMPLES']
        }
    
    ## list aggregation
    list_aggregated = defaultdict(list)
    
    for d in perm_choice_dict_processed.values():
        curr_non_target = []
        norm_const = sum(d.values())
        
        for k,v in d.items():
            if k == 'TARGET':
                list_aggregated[k] += [v/norm_const]
            else:
                curr_non_target += [v/norm_const]
        list_aggregated['NON_TARGET'] += [sum(curr_non_target)]
    
    ## Standard Error of the Mean
    sem = {
        k+"_SEM": stats.sem(v) 
        for k,v in list_aggregated.items()
        }
    
    ## Standard Deviation
    std = {
        k+"_STD": np.std(v) 
        for k,v in list_aggregated.items()
        }
    
    return dict(sum_aggregated | totals | aggregated_norm | sem | std)
        
def get_exp_aggregation(results, simulated_sample_size=100):
    exp_results = ut.read_results(results)
    exp_results["permutation_choices"] = exp_results.apply(lambda x: (x.permutation,x.choices), axis=1)
    
    exp_agg = exp_results.groupby(["llm", "condition"])\
                         .agg({
                             'permutation_choices': lambda x: aggregate_over_permutations(
                                 dict(list(x)), # permutation-choice dictioanry
                                 simulated_sample_size=simulated_sample_size
                                 )
                             })
    
    aggregated_df = exp_agg['permutation_choices'].apply(pd.Series).reset_index(inplace=False).fillna(0.)
    aggregated_df['job_opening'] = exp_results.loc[0]['exp_params']['job_opening']
    
    return aggregated_df

def agg_results_analysis(result_files):
    agg_dfs = []

    for res_f_name_path in result_files:
        exp_aggr_df = get_exp_aggregation(res_f_name_path)
        agg_dfs += [exp_aggr_df]

    agg_concat_dfs = pd.concat(agg_dfs).reset_index(drop=True)

    return agg_concat_dfs

# PLOTTING
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    
    callback(axes)
    
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    
    return axes

def plot_target_perm_probs(results, ax=None, x="CONTROL", y="DECOY", show_legend=True):
    exp_results_pd = ut.read_results(results)
    # job_title = exp_results_pd.loc[0,'job_details']["job_title"]
    job_title = exp_results_pd.loc[0,'exp_params']['job_details']["job_title"]
    exp_results_pd['target_prob'] = exp_results_pd['choices'].apply(lambda x: x.get('TARGET', 0)/sum(x.values()))

    pd_df = pd.pivot_table(
        exp_results_pd[['llm','condition','target_prob','permutation']], 
        index=['llm', 'permutation'], 
        columns=['condition'], 
        values='target_prob'
        ).reset_index()

    colors = {
        llm_runner.models.GPT35.value:'blue', 
        llm_runner.models.GPT4.value: 'red'
        }

    color_list = [
        colors[group] 
        for group in pd_df['llm']
        ]
    
    ax = pd_df.plot.scatter(ax=ax, x=x, y=y, c=color_list)
    
    for _, row in pd_df.iterrows():
        ax.annotate(
            row['permutation'], 
            (row[x], row[y]), 
            xytext=(0,3),
            textcoords='offset points', 
            family='sans-serif', 
            fontsize=10
            )
        
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_xlabel("P(TARGET|CONTROL)")
    ax.set_ylabel("P(TARGET|TREATMENT)")
    # ax.set_box_aspect(1)
    ax.set_title(job_title)

    legend_handles = [
        Patch(color=colors[llm_runner.models.GPT35.value], label=model_display_names[llm_runner.models.GPT35.value]),
        Patch(color=colors[llm_runner.models.GPT4.value], label=model_display_names[llm_runner.models.GPT4.value]),
    ]
    
    if show_legend:
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            # bbox_to_anchor=(1.04, 1), 
            )

    add_identity(ax, color='gray', ls='--')

    return ax

def plot_target_perm_probs_over_jobs(result_files, tag="", figsize=(9,6)):
    fig, axs = plt.subplots(
        nrows=2, 
        ncols=3, 
        figsize=figsize, 
        squeeze=True, 
        sharey=True, 
        sharex=True
        )
    
    # fig.suptitle('TARGET choice under candidate presentation order permutations', fontsize=18)

    for i, res_f_name_path in enumerate(result_files):
        plot_target_perm_probs(
            res_f_name_path, 
            axs.flatten()[i], 
            show_legend=i==2
            )
        
    plt.tight_layout()

    fig.savefig(f"../results/figures/{tag}_target_choice_perm.pdf")

    # return fig


# EXPERIMENTS
def analyse_results_contr_treat(result_files, simulated_sample_size=100, significance_threshold=0.05):
    results = []

    for res_f_name_path in result_files:

        exp_results = ut.read_results(res_f_name_path)
        
        exp_tag = exp_results['tag'].to_list()[0].split("_")[-1]

        data_agg_over_perms = get_exp_aggregation(exp_results, simulated_sample_size=simulated_sample_size)
        
        for curr_model in llm_runner.get_models(value=True):
            curr_model_data_agg_over_perms = data_agg_over_perms[data_agg_over_perms["llm"]==curr_model].copy()
            
            # skip analysis if data for the model is missing
            if curr_model_data_agg_over_perms.empty: continue  
            
            # bias magnitude
            control_target_prob = curr_model_data_agg_over_perms[curr_model_data_agg_over_perms['condition']=='CONTROL']["TARGET_AVG_PROB"].values[0]
            curr_model_data_agg_over_perms["bias_score"] = curr_model_data_agg_over_perms["TARGET_AVG_PROB"] - control_target_prob
            
            # pd_exp_agg_model = curr_model_data_agg_over_perms[curr_model_data_agg_over_perms['condition'] != 'CONTROL']["bias_score"].values[0]  # ASSUMPTION: only one cont

            # chi square
            chi_square_columns = ['condition', 'TARGET', 'NON_TARGET']
            model_contingency_table = curr_model_data_agg_over_perms[chi_square_columns]\
                                        .set_index("condition", drop=True, inplace=False)
            
            chi_square = stats.chi2_contingency(model_contingency_table)
            
            chi_square_stats = {
                "chi^2_statistic": chi_square.statistic,
                "chi^2_pvalue": chi_square.pvalue,
                "significant": chi_square.pvalue<significance_threshold,
                "chi^2_dof": chi_square.dof,
            }

            # avg probs and errors
            avg_prob_errors_columns = ['condition', 'TARGET_AVG_PROB', 'TARGET_SEM', 'TARGET_STD', 'bias_score']

            avg_error = curr_model_data_agg_over_perms[avg_prob_errors_columns].set_index("condition", drop=True, inplace=False).to_dict(orient='index')
            
            avg_error_flat = {
                f"{k}_{kk}":vv 
                for k,v in avg_error.items() 
                for kk,vv in v.items()
                }
            
            results += [{
                "model": curr_model,
                "experiment": exp_tag,
                **chi_square_stats,
                **avg_error_flat
            }]
            
    return results

def plot_results_contr_treat(results, tag="", figsize=(10, 5), wrap_char_length=10, save=True):
    width = 0.5  # the width of the bars

    llms = set([d['model'] for d in results])

    llm_number = len(llms)
    
    fig, axs = plt.subplots(llm_number, 1, figsize=figsize, sharex=True, constrained_layout=True)

    axes = [axs] if llm_number == 1 else axs
    
    for curr_model, ax in zip(sorted(llms), axes):
        model_results = [r for r in results if r['model']==curr_model]
        
        res_df = pd.DataFrame(model_results).drop(columns=['model'])
        
        res_df['experiment'] = pd.Categorical(
            res_df['experiment'],
            categories=pl.jt_order_female_dominated + pl.jt_order_male_dominated, 
            ordered=True)
        
        res_df = res_df.sort_values(by=['experiment'])
        
        control_means, control_std = res_df['CONTROL_TARGET_AVG_PROB'], res_df['CONTROL_TARGET_SEM']
        treatment_means, treatment_std = res_df['DECOY_TARGET_AVG_PROB'], res_df['DECOY_TARGET_SEM']
        
        ind = np.arange(len(control_means)*2, step=2)  # the x locations for the groups
        
        rects_control = ax.bar(
            ind - width/2, 
            control_means, 
            width, 
            yerr=control_std,
            label='CONTROL',
            color="#007894"  # teal
            )
        
        rects_treatment = ax.bar(
            ind + width/2, 
            treatment_means, 
            width, 
            yerr=treatment_std,
            label='TREATMENT',
            color="goldenrod"  # golden-yellow
            )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylim((0,1))
        ax.set_ylabel("$\mathrm{P}(TARGET)$")
        ax.set_title(f'Model: {model_display_names[curr_model]}')#, fontweight='bold')
        ax.set_xticks(ind)
        
        ax.set_xticklabels([
            ut.wrap_string(x,wrap_char_length) 
            for x in res_df['experiment'].to_list()
            ])
        
        ax.axhline(y=0.5, c='grey', linestyle='dashed', zorder=0)


    axes[0].legend(loc=(0.3,0.65))
    
    if save:
        fig.savefig(f"../results/figures/{tag}_decoy_effect_bar.pdf")
    
    return fig

# EXP 2 
def plot_bias_space(result_files, tag="", model_filter=ModelDeployments.GPT35.value, figsize=(12, 8)):
    def prep_data(res_f_name_path):
        pd_exp = ut.read_results(res_f_name_path)
        alt_defs = pd_exp[["TARGET", "COMPETITOR"]].iloc[0].to_dict()
        job_title = pd_exp['exp_params'][0]['job_details']["job_title"]

        def get_axis_label(axis='dim_1'):
            dim_data = pd_exp['exp_params'][0]['job_details'][axis]
            return f"{dim_data['label']} {dim_data['type']} [{dim_data['unit']}]"
        
        # prepare dataframe for visualisation
        pd_exp['condition'] = pd_exp['condition'].map(str)
        pd_exp_agg = get_exp_aggregation(pd_exp)
        pd_exp_agg_model = pd_exp_agg[pd_exp_agg["llm"] == model_filter].copy()
        control_target_prob = pd_exp_agg_model[pd_exp_agg_model['condition']=='CONTROL']["TARGET_AVG_PROB"].values[0]
        
        pd_exp_agg_model["bias_score"] = pd_exp_agg_model["TARGET_AVG_PROB"] - control_target_prob
        pd_exp_agg_model = pd_exp_agg_model[pd_exp_agg_model['condition'] != 'CONTROL']
        pd_exp_agg_model['condition'] = pd_exp_agg_model['condition'].map(ast.literal_eval)
        
        bias_coords = pd_exp_agg_model[["condition", "bias_score"]].copy()
        bias_coords[['dim_1_value','dim_2_value']] = pd.DataFrame(bias_coords.condition.tolist(), index=bias_coords.index)
        bias_coords = bias_coords.drop(columns=['condition'])

        bias_coords_pivot = bias_coords.pivot(index='dim_2_value', 
                                              columns='dim_1_value', 
                                              values='bias_score')\
                                        .sort_index(ascending=False)  # ASSUMPTION: index, i.e., dim_2 is numerical, while columns are catergorical
        
        # order columns (categorical dimension)
        dim_1_ordered_values = pd_exp['exp_params'][0]['job_details']['dim_1']['values']
        bias_coords_pivot = bias_coords_pivot[dim_1_ordered_values]
        
        bias_coords_pivot_fillna = bias_coords_pivot.fillna(0.)

        bias_coords_pivot_labels = bias_coords_pivot.astype(str, copy=True)
        bias_coords_pivot_labels.loc[:] = ""
        
        for a in ["TARGET", "COMPETITOR"]:
            bias_coords_pivot_labels.loc[
                alt_defs[a]['dim_2_value'],
                alt_defs[a]['dim_1_value'],
                ] = a[0]
        
        bias_interval = bias_coords["bias_score"].min(), bias_coords["bias_score"].max()
        
        return {
            "job_title": job_title,
            "alt_defs": alt_defs,
            "bias_coords_pivot_labels": bias_coords_pivot_labels,
            "bias_interval": bias_interval,
            "bias_coords_pivot_fillna": bias_coords_pivot_fillna,
            "dim_1_ordered_values": dim_1_ordered_values,
            "get_axis_label": get_axis_label,
            }
    
    # prep data for visualisation
    bias_interval = (0,0)
    prepped_data = []
    
    for res_f_name_path in result_files:
        d = prep_data(res_f_name_path)
        prepped_data += [d]
        bias_interval_curr = d['bias_interval']
        
        bias_interval = (
            bias_interval_curr[0] if bias_interval_curr[0]<bias_interval[0] else bias_interval[0],
            bias_interval_curr[1] if bias_interval_curr[1]>bias_interval[1] else bias_interval[1],
            )
    
    # visualise
    fig, axs = plt.subplots(2, 3, figsize=figsize, squeeze=True, sharey=True, sharex='col')
    fig.suptitle(f'Model: {model_display_names[model_filter]}', fontsize=18)#, fontweight='bold')
    cbar_ax = fig.add_axes([1., .3, .02, .4])

    jt_indices = {
        **{jt:(0,i) for i,jt in enumerate(pl.jt_order_female_dominated)},
        **{jt:(1,i) for i,jt in enumerate(pl.jt_order_male_dominated)},
        }

    for d in prepped_data:
        # is_x_categorical = d['job_title'] in ["Social Psychologist", "Mechanical engineer"]
        
        ax = sns.heatmap(
            ax=axs[
                jt_indices[d['job_title']][0],
                jt_indices[d['job_title']][1]
                ],
            data=d['bias_coords_pivot_fillna'],
            annot=d['bias_coords_pivot_labels'], 
            center=0,
            fmt="",
            annot_kws={
                'fontsize':24, 
                'color':'black', 
                'verticalalignment':'center', 
                'horizontalalignment':'center'
                },
            cmap='RdBu_r',
            # square=not is_x_categorical,
            vmin=bias_interval[0],
            vmax=bias_interval[1],
            cbar=jt_indices[d['job_title']]==(0,0),
            cbar_ax=None if jt_indices[d['job_title']]!=(0,0) else cbar_ax,
            cbar_kws=dict(label="Bias"),
            )

        # if is_x_categorical:
        #     ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right')  # ha='right' for right alignment

        # TODO: generalise rectangle drawing
        # max_dim_value = max(bias_coords_pivot_fillna.index.values)
        max_dim_value = len(d['bias_coords_pivot_fillna'].index.values)
        
        for a in ["TARGET", "COMPETITOR"]:
            x_coord = (
                max_dim_value-d['alt_defs'][a]['dim_1_value'] 
                if isinstance(d['alt_defs'][a]['dim_1_value'], int)
                else d['dim_1_ordered_values'].index(d['alt_defs'][a]['dim_1_value'])+1
                )
            
            ax.add_patch(
                Rectangle(
                    xy=(x_coord, d['alt_defs'][a]['dim_2_value']), 
                    width=1, 
                    height=-1, 
                    fill=True, 
                    color='white',
                    # edgecolor='white', 
                    lw=0,
                    )
                )
        
        axis_label_font_size = 12.5
        
        ax.set_xlabel(d['get_axis_label'](axis='dim_1'), fontdict={'size': axis_label_font_size})  # , 'weight': 'bold'
        ax.set_ylabel(d['get_axis_label'](axis='dim_2'), fontdict={'size': axis_label_font_size})
        
        ax.set_title(d['job_title'], fontsize=18)
    
    cbar_ax.yaxis.label.set_size(18)

    plt.tight_layout()
    
    plt.show(fig)

    fig.savefig(f"../results/figures/{tag}_space_{model_filter}.pdf", bbox_inches='tight')
    return fig

# EXP 3
def vis_gender_effect(anaysed_df, tag="", figure_size=(5,5)):
    col_rename_dict = {
        'experiment': "Job titles", 
        'model': "Model", 
        'DECOY_bias_score': "Bias", 
        'target_pronoun': "TARGET",
        'competitor_pronoun': "COMPETITOR", 
        'context_pronoun': "DECOY",
        }

    pronoun_map = {
        "his": "male",
        "her": "female",
        }

    dfs = []

    for (target_pronoun, competitor_pronoun, context_pronoun), results in anaysed_df.items():
        df = pd.DataFrame(results)[["model", "DECOY_bias_score", "experiment"]].set_index(["experiment", "model"])
        
        df[["target_pronoun", "competitor_pronoun", "context_pronoun"]] = (
            pronoun_map[target_pronoun], 
            pronoun_map[competitor_pronoun], 
            pronoun_map[context_pronoun]
            )
        
        dfs += [df]

    dfs_concat = pd.concat(dfs, axis=0)\
                    .reset_index(drop=False)\
                    .rename(columns=col_rename_dict)
                    

    dfs_concat = dfs_concat.replace({col_rename_dict["model"]: model_display_names})
    sns.set_theme(rc={'figure.figsize':figure_size})
    sns.set_style('whitegrid', {
        'font.family':'serif', 
        'font.serif':'Times New Roman'
        })



    graph = sns.FacetGrid(
        dfs_concat,
        col=col_rename_dict["model"],
        row=col_rename_dict["target_pronoun"], 
        hue=col_rename_dict["experiment"],
        hue_order=pl.jt_order_female_dominated+pl.jt_order_male_dominated,
        aspect=1,
        margin_titles=True,
        )

    graph.set_titles(col_template="{col_var}: {col_name}",row_template="{row_var}: {row_name}")

    graph.map(sns.pointplot, 
            col_rename_dict["context_pronoun"], 
            col_rename_dict["DECOY_bias_score"],
            order=["male", "female"],
            ).add_legend(title="Occupations")#"Job titles")
    
    plt.savefig(f"../results/figures/{tag}_gender_effect.pdf", bbox_inches='tight')
    
    return dfs_concat


def gender_effect_significance(df, reset_index=False, significance_threshold=0.05):
    grouped = df.groupby(["Model", "TARGET", "DECOY"]).agg(list)

    # assert identical order for the paired test
    frozensets_jobs = grouped["Job titles"].apply(frozenset)
    assert frozensets_jobs.nunique() == 1, "Not all lists in the column are the same."


    df_significance = grouped.groupby(["Model", "TARGET"])['Bias']\
                            .apply(
        lambda x: stats.ttest_rel(x.iloc[0], x.iloc[1], alternative="two-sided"))\
                            .apply(pd.Series)

    df_significance.columns = ['t-statistic', 'p-value']
    
    df_significance['significant'] = df_significance['p-value']<significance_threshold
    
    if reset_index:
        df_significance = df_significance.reset_index()
    
    return df_significance

# EXP 4
def vis_decoy_explanation_effect(anaysed_df, tag="", figure_size=(5,5)):
    col_rename_dict = {
        'experiment': "Job titles", 
        'model': "Model", 
        'DECOY_bias_score': "Bias", 
        'decoy_explanation': "Warning",
        }

    dfs = []

    for decoy_explanation, results in anaysed_df.items():
        df = pd.DataFrame(results)[["model", "DECOY_bias_score", "experiment"]].set_index(["experiment", "model"])
        df["decoy_explanation"] = decoy_explanation
        dfs += [df]

    dfs_concat = pd.concat(dfs, axis=0)\
                    .reset_index(drop=False)\
                    .rename(columns=col_rename_dict)
                    
    dfs_concat = dfs_concat.replace({col_rename_dict["model"]: model_display_names})

    sns.set_theme(rc={'figure.figsize':figure_size})

    sns.set_style('whitegrid', {
        'font.family':'serif', 
        'font.serif':'Times New Roman'
        })

    graph = sns.FacetGrid(
        dfs_concat,
        # col=col_rename_dict["model"],
        row=col_rename_dict["model"],
        hue=col_rename_dict["experiment"],
        hue_order=pl.jt_order_female_dominated+pl.jt_order_male_dominated,
        aspect=1.6,
        margin_titles=True,
        )

    graph.set_titles(
        col_template="{col_var}: {col_name}",
        row_template="{row_var}: {row_name}"
        )

    graph.map(sns.pointplot, 
            col_rename_dict["decoy_explanation"], 
            col_rename_dict["DECOY_bias_score"],
            order=["Absent", "Present"],
            ).add_legend(title="Occupations")#"Job titles")
    
    plt.savefig(f"../results/figures/{tag}_decoy_explanation_effect.pdf", bbox_inches='tight')
    
    return dfs_concat

def decoy_explanation_effect_significance(df, reset_index=False, significance_threshold=0.05):
    grouped = df.groupby(["Model", "Warning"]).agg(list)
    
    # assert identical order for the paired test
    frozensets_jobs = grouped["Job titles"].apply(frozenset)
    assert frozensets_jobs.nunique() == 1, "Not all lists in the column are the same."

    df_significance = grouped.groupby(["Model"])['Bias']\
                            .apply(
                                lambda x: stats.ttest_rel(x.iloc[0], x.iloc[1], alternative="two-sided")
                                )\
                            .apply(pd.Series)
    
    # display(grouped.groupby(["Model"])['Bias']\
    #                         .apply(
    #                             lambda x: stats.ttest_rel(x.iloc[0], x.iloc[1], alternative="two-sided")
    #                             ))
    
    df_significance.columns = ['t-statistic', 'p-value']
    
    df_significance['significant'] = df_significance['p-value']<significance_threshold
    
    if reset_index:
        df_significance = df_significance.reset_index()
    
    return df_significance

# EXP 5
def vis_instruction_variation_effect(anaysed_df, tag="", figure_size=(5,5)):
    col_rename_dict = {
        'experiment': "Job titles", 
        'model': "Model", 
        'DECOY_bias_score': "Bias", 
        'instruction_variation': "Recruiter role definition",
        }

    dfs = []

    for instruction_variation, results in anaysed_df.items():
        df = pd.DataFrame(results)[["model", "DECOY_bias_score", "experiment"]].set_index(["experiment", "model"])
        df["instruction_variation"] = instruction_variation
        dfs += [df]

    dfs_concat = pd.concat(dfs, axis=0)\
                    .reset_index(drop=False)\
                    .rename(columns=col_rename_dict)
                    
    dfs_concat = dfs_concat.replace({col_rename_dict["model"]: model_display_names})

    sns.set_theme(rc={'figure.figsize':figure_size})

    sns.set_style('whitegrid', {
        'font.family':'serif', 
        'font.serif':'Times New Roman'
        })

    graph = sns.FacetGrid(
        dfs_concat,
        # col=col_rename_dict["model"],
        row=col_rename_dict["model"],
        hue=col_rename_dict["experiment"],
        hue_order=pl.jt_order_female_dominated+pl.jt_order_male_dominated,
        aspect=1.8,
        margin_titles=True,
        )

    graph.set_titles(
        col_template="{col_var}: {col_name}",
        row_template="{row_var}: {row_name}"
        )

    graph.map(sns.pointplot, 
            col_rename_dict["instruction_variation"], 
            col_rename_dict["DECOY_bias_score"],
            order=["Succinct", "Concise-1", "Concise-2", "Verbose"],
            ).add_legend(title="Occupations")#"Job titles")

    plt.savefig(f"../results/figures/{tag}_instruction_variation_effect.pdf", bbox_inches='tight')
    
    return dfs_concat

from statsmodels.stats.anova import AnovaRM 

def instruction_variation_effect_significance(df, significance_threshold=0.05):
    models = df["Model"].drop_duplicates().to_list()
    
    renaming_dict = {
        t:t.replace(" ", "_")
        for t in ['Recruiter role definition','Job titles']
        }

    significance_per_model = {}
    anova_tables = []
    for model in models:
        df_model = df[df["Model"]==model].copy()
        
        # AnovaRM does not line space in column names
        df_model = df_model.rename(columns=renaming_dict)  

        anova_rm = AnovaRM(
            data=df_model, 
            depvar='Bias',
            subject=renaming_dict["Job titles"], 
            within=[renaming_dict['Recruiter role definition']]
            ).fit()
        
        anova_table = anova_rm.anova_table
        anova_table["Model"] = model
        anova_table["significant"] = anova_table['Pr > F']<significance_threshold

        anova_tables += [anova_table]
        
        f_value = anova_table.loc[renaming_dict['Recruiter role definition'], 'F Value']
        num_df = anova_table.loc[renaming_dict['Recruiter role definition'], 'Num DF']
        den_df = anova_table.loc[renaming_dict['Recruiter role definition'], 'Den DF']
        p = anova_table.loc[renaming_dict['Recruiter role definition'], 'Pr > F']
        sign = '<' if p < significance_threshold else '>'
        
        significance_per_model[model] = f"F({num_df}, {den_df}) = {f_value}, p {sign} {significance_threshold}"

    return pd.concat(anova_tables, axis=0), significance_per_model