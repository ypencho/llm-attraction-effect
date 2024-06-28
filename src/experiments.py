from llm_runner import LLMRunner
from alternatives import Alternatives
import prompt_library as pl
import utils as ut

import json
import pandas as pd
from tqdm import tqdm
from itertools import product

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


def is_phantom(line_start, line_end, point):
    """Decide if an alternative should be a phantom one. 
    Find area of the triangle formed by the line and the point. Phantom status determined by sign if area.
    """
    area = (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - \
           (point[0] - line_start[0]) * (line_end[1] - line_start[1])

    if area > 0:
        return False  # Point is to the left of the line -> Not phantom
    elif area < 0:
        return True  # Point is to the right of the line -> Phantom
    else:
        return False  # Point lies on the indifference line -> Not phantom


def generate_context_grid(target_competitor, dim_1_values, dim_2_values):
    """Enumerate all possible context candidate profiles.
    Note: Correct phantom status is assigned when TARGET's dim_1_value > COMPETITOR's dim_1_value; flip line_start and line_end when the opposite holds
    """
    # replace category levels with indices
    is_categorical_dim = lambda dim_values: all(isinstance(v, str) for v in dim_values)
    get_cat_id_dict = lambda dim_values: {v:i for i,v in enumerate(dim_values)} if is_categorical_dim(dim_values) else {v:v for v in dim_values}
    convert_coords = lambda tuple, d0, d1:(d0[tuple[0]], d1[tuple[1]])
    
    dim_1_cat_dict = get_cat_id_dict(dim_1_values)
    dim_2_cat_dict = get_cat_id_dict(dim_2_values)
    
    grid = list(product(dim_1_values, dim_2_values))
    
    t_c = [
        (
            target_competitor["TARGET"]["dim_1_value"], 
            target_competitor["TARGET"]["dim_2_value"]
            ),
        (
            target_competitor["COMPETITOR"]["dim_1_value"], 
            target_competitor["COMPETITOR"]["dim_2_value"]
            ),
        ]

    # phantom half-plane    
    target_coords = convert_coords(t_c[0],dim_1_cat_dict,dim_2_cat_dict)
    competitor_coords = convert_coords(t_c[1],dim_1_cat_dict,dim_2_cat_dict)
    
    line_start, line_end = (competitor_coords, target_coords) if target_coords[0] < competitor_coords[0] else (target_coords, competitor_coords)
    
    alt_defs_context_grid = lambda target_pronoun,competitor_pronoun,context_pronoun: {
        "TARGET": {
            **target_competitor["TARGET"],
            "pronoun": target_pronoun,
            },
        "COMPETITOR": {
            **target_competitor["COMPETITOR"],
            "pronoun": competitor_pronoun,
            },
        "CONTEXT": {
            t:{
                "dim_1_value": t[0], 
                "dim_2_value": t[1],
                "pronoun": context_pronoun,
                "phantom": is_phantom(
                    line_start=line_start,
                    line_end=line_end,
                    point=convert_coords(t,dim_1_cat_dict,dim_2_cat_dict)),
                }
            for t in grid if t not in t_c
            }
        }
    
    return alt_defs_context_grid


class ContextExperiment:
    def __init__(self, 
                 target, competitor, context, 
                 prompt_template, candidate_template, 
                 candidate_phantom_state, job_phantom_requirement, assume_phantom_in_choice_set=False,
                 tag="", description=""):
        self.target = target
        self.competitor = competitor
        self.context = context
        
        self.prompt_template = prompt_template
        self.candidate_template = candidate_template
        
        self.candidate_phantom_state = candidate_phantom_state
        self.job_phantom_requirement = job_phantom_requirement
        self.assume_phantom_in_choice_set = assume_phantom_in_choice_set
        self.tag = tag
        self.description = description

        self.llm_runner = LLMRunner()
        
    def get_exp_as_pd(self, cat_labels=None):
        df = pd.DataFrame.from_dict({"TARGET":self.target, 
                                    "COMPETITOR":self.competitor, 
                                    **self.context})\
                        .transpose()\
                        .infer_objects()
                        
        if cat_labels:  # check for a categorical column -> change dtype of categorical columns
            cat_columns = [
                column
                for column in df.columns
                if all(entry in cat_labels for entry in df[column])
                ]
            
            for cat_col in cat_columns:
                df[cat_col] = pd.Categorical(df[cat_col], categories=cat_labels , ordered=True)
                df = df.sort_values(cat_col)
        
        return df
    
    def plot_exp(self, x_label=None, y_label=None, plot_title=None, c="phantom", cat_labels=None, figsize=(5,5), xytext=(0,0)):
        x = "dim_1_value"
        y = "dim_2_value"
        
        exp_pd_df = self.get_exp_as_pd(cat_labels=cat_labels)
        
        if c:
            if exp_pd_df[c].dtype == 'bool':
                exp_pd_df[c] = exp_pd_df[c].astype(int)

        ax = exp_pd_df.plot.scatter(x=x, y=y, c=c, colormap='viridis', figsize=figsize)

        for idx, row in exp_pd_df.iterrows():
            ax.annotate(
                idx, 
                (row[x], row[y]), 
                xytext=xytext,
                textcoords='offset points', 
                family='sans-serif', 
                fontsize=10
                )

        if x_label: ax.set_xlabel(x_label)
        if x_label: ax.set_ylabel(y_label)
        if plot_title: ax.set_title(plot_title)

        return ax

    def get_control(self, phantom_key="phantom"):
        phantom_in_choice_set = self._exists_phantom_in_choice_set(self.context, phantom_key=phantom_key)

        # control cannot include a target or competitor phantom candidate, but a phantom candidate ca be present in the corresponsing treatment
        phantom_state = self.candidate_phantom_state[phantom_in_choice_set]['not_phantom']
        
        candidates = {
            "TARGET": self.candidate_template.format(**self.target, phantom_state=phantom_state), 
            "COMPETITOR": self.candidate_template.format(**self.competitor, phantom_state=phantom_state),
            }
 
        selection_prompt = self.prompt_template.format(phantom_requirement=self.job_phantom_requirement[phantom_in_choice_set])
        
        return {
            "prompt_template": selection_prompt,
            "candidates": candidates,
        }
    
    def _exists_phantom_in_choice_set(self, context_dict, phantom_key="phantom"):
        if self.assume_phantom_in_choice_set:
            return True
        
        return any([c.get(phantom_key, False) for c in context_dict.values()])
    
    def get_treatments(self, phantom_key="phantom"):
        phantom_in_choice_set = self._exists_phantom_in_choice_set(self.context, phantom_key=phantom_key)
        treatments = {}

        for k,v in self.context.items():
            control = self.get_control()
            
            phantom_state = (
                self.candidate_phantom_state[phantom_in_choice_set]['phantom'] 
                if v.get(phantom_key, False) 
                else self.candidate_phantom_state[phantom_in_choice_set]['not_phantom']
                )

            control['candidates']['CONTEXT'] = self.candidate_template.format(**v, phantom_state=phantom_state)

            treatments[k] = control

        return treatments
    
    def get_all_experiment_instances(self, enum, phantom_key="phantom"):
        substitute_candidate_names = lambda curr_enum, candidates: {
            curr_enum[k].value:v.format(cand_name=curr_enum[k].value) 
            for k,v in candidates.items()
            }
        
        def _assemble_prompt_enum(curr_enum, name_vocabulary, prompt_template, candidates):
            curr_candidates = substitute_candidate_names(curr_enum, candidates)
            
            curr_candidates_prompt = "\n".join([
                curr_candidates[name]
                for name in name_vocabulary
                if name in curr_candidates
                ])

            curr_name_vocab_prompt = ", ".join([
                name
                for name in name_vocabulary
                if name in curr_candidates
                ])

            curr_prompt = prompt_template.format(
                candidates=curr_candidates_prompt,
                alternatives=curr_name_vocab_prompt
            )

            return curr_prompt
        
        all_experiments = {}
        
        # get control prompt
        control = self.get_control(phantom_key=phantom_key)
        
        control_prompt = _assemble_prompt_enum(
            curr_enum=enum['control_enum'], 
            name_vocabulary=enum['name_vocabulary'], 
            prompt_template=control['prompt_template'], 
            candidates=control['candidates']
            )
        
        all_experiments['CONTROL'] = {
            "prompt": control_prompt,
            "choice_decoder": enum['control_enum']
        }
        
        # get treatment prompts
        treatments = self.get_treatments(phantom_key=phantom_key)
        
        for treatment_name, treatment in treatments.items():
            treatment_prompt = _assemble_prompt_enum(
                curr_enum=enum['treatment_enum'], 
                name_vocabulary=enum['name_vocabulary'], 
                prompt_template=treatment['prompt_template'], 
                candidates=treatment['candidates']
                )
            
            all_experiments[treatment_name] = {
                "prompt": treatment_prompt,
                "choice_decoder": enum['treatment_enum']
            }
            
        return all_experiments
    
    def run(self, name_vocabulary=["A", "B", "C"], llm_models=None, permutations=True, temperature=1., num_samples=2, phantom_key="phantom", metadata={}):
        llms = llm_models if llm_models else self.llm_runner.get_models()
        
        alts = Alternatives(name_vocabulary=name_vocabulary)
        enums_alternatives = alts.get_enums(permutations=permutations)

        exp_instances_perms = [
            self.get_all_experiment_instances(alt_permutation, phantom_key=phantom_key)
            for alt_permutation in enums_alternatives
            ]
        
        results = []
        
        phantom_in_choice_set = self._exists_phantom_in_choice_set(self.context, phantom_key=phantom_key)
        
        for i, perm in enumerate(tqdm(exp_instances_perms, desc='Presentation permutations', position=0)):
            for llm in llms:
                choice_procedure = 'sample' if llm == self.llm_runner.models.GPT4 else 'logprobs'
                
                for condition, condition_def in perm.items():
                    
                    if llm == self.llm_runner.models.GPT35:
                        choices = self.llm_runner.choice_logprobs_gpt35(
                            prompt=condition_def['prompt'], 
                            decoding_enum=condition_def['choice_decoder'],
                            temperature=temperature
                            )
                    if llm == self.llm_runner.models.GPT4:
                        choices = self.llm_runner.choice_sample_gpt4(
                            prompt=condition_def['prompt'], 
                            decoding_enum=condition_def['choice_decoder'],
                            temperature=temperature,
                            num_samples=num_samples
                            )

                    results += [{
                        **metadata,
                        "tag": self.tag,
                        "permutation": i,
                        "llm": llm.value,
                        "condition": condition,
                        "prompt": condition_def['prompt'],
                        "choice_decoder": {
                            a.name:a.value 
                            for a in condition_def['choice_decoder']
                            },
                        "choice_procedure": choice_procedure,
                        "temperature": temperature,
                        "choices": choices,
                        "phantom_in_choice_set": phantom_in_choice_set,
                    }]
        
        return results
    
    
def get_alt_defs(job_details, exp_params, gen_grid=False):
    if gen_grid:
        dim_num_values = exp_params['dim_num_values']
        dim_cat_values = exp_params['dim_cat_values']
        
        alt_defs_fcn, dim_1_values, dim_2_values = (
            (pl.num_num_target_competitor_params, dim_num_values, dim_num_values)    # ASSUMPTION: identical ranges for dim 1 and dim 2
            if pl.num_num in job_details['tags'] 
            else (pl.cat_num_target_competitor_params, dim_cat_values, dim_num_values)  # ASSUMPTION: dim 1 is categorical and dim 2 is numerical
            )

        alt_defs = generate_context_grid(
            target_competitor=alt_defs_fcn(target_pronoun="", competitor_pronoun=""),
            dim_1_values=dim_1_values,
            dim_2_values=dim_2_values,
            )(
                target_pronoun=exp_params["target_pronoun"],
                competitor_pronoun=exp_params["competitor_pronoun"],
                context_pronoun=exp_params["context_pronoun"],
            )
    else:
        alt_defs_fcn = pl.get_num_num_alt_defs if pl.num_num in job_details['tags'] else pl.get_cat_num_alt_defs

        # Parametrisation of alternatives
        alt_defs = alt_defs_fcn(
            context=exp_params["context"],
            target_pronoun=exp_params["target_pronoun"],
            competitor_pronoun=exp_params["competitor_pronoun"],
            context_pronoun=exp_params["context_pronoun"],
            )
        
    return alt_defs

def run_exp_for_job(exp_params_0, job_title, job_details, llm_models=None, gen_grid=False, c=None):
    exp_params = dict(
        **exp_params_0,
        job_opening=job_title,
        job_details=job_details,
        tag=exp_params_0["tag_base"] + "_" + job_title,
        )
    
    alt_defs = get_alt_defs(job_details, exp_params, gen_grid=gen_grid)

    # Job prompt and candidate templates assembly
    pr_templ, cand_templ = pl.job_prompt_assembler(
        recruiter_instruction=exp_params["recruiter_instruction"],
        decoy_explanation=exp_params["decoy_explanation"],
        job_opening=job_details,
        )

    exp = ContextExperiment(
        target=alt_defs["TARGET"], 
        competitor=alt_defs["COMPETITOR"],
        context=alt_defs["CONTEXT"], 
        prompt_template=pr_templ,
        candidate_template=cand_templ,
        candidate_phantom_state=exp_params["candidate_phantom_state"],
        job_phantom_requirement=exp_params["job_phantom_requirement"],
        assume_phantom_in_choice_set=exp_params.get("assume_phantom_in_choice_set", False),
        tag=exp_params["tag"],
        description=exp_params["description"]
        )

    exp_results = exp.run(
        name_vocabulary=exp_params["name_vocabulary"], 
        permutations=True, 
        temperature=exp_params["temperature"],
        num_samples=exp_params["num_samples"],
        phantom_key="phantom", 
        metadata={
            "exp_params":exp_params, 
            "TARGET":alt_defs["TARGET"],
            "COMPETITOR":alt_defs["COMPETITOR"],
            },
        llm_models=llm_models,
        )

    exp_results_file_name = f"../results/data/{exp.tag}_choices.json"

    with open(exp_results_file_name, 'w') as f:
        json.dump(exp_results, f)
        print('Saved:', exp_results_file_name)
    
    # 2D plot of the experimental setting
    exp.plot_exp(
        x_label=ut.get_axis_label(job_details, axis='dim_1'), 
        y_label=ut.get_axis_label(job_details, axis='dim_2'), 
        plot_title=job_title,
        c=c,
        cat_labels=pl.educational_degrees_ordered,
        xytext=(-10,5),
        )
    
    return exp_results_file_name
    
if __name__ == "__main__":
    # PHANTOM
    line_start = (7, 5)  # target
    line_end = (5, 7)  # competitor
    point1 = (10, 2)
    point2 = (8, 6)
    point3 = (6, 4)

    print("Point 1 is:", is_phantom(line_start, line_end, point1))
    print("Point 2 is:", is_phantom(line_start, line_end, point2))
    print("Point 3 is:", is_phantom(line_start, line_end, point3))
        
    dim_values = lambda dim_range: range(dim_range[0], dim_range[1]+1)

    # CONTEXT GRID
    from prompt_library import educational_degrees_ordered

    ## Classical definition
    alt_defs_numerical_numerical = lambda target_pronoun,competitor_pronoun,context_pronoun: {
        "TARGET": {
            "dim_1_value": 7,  # y
            "dim_2_value": 5,  # x
            "pronoun": target_pronoun,
            },
        "COMPETITOR": {
            "dim_1_value": 5, 
            "dim_2_value": 7,
            "pronoun": competitor_pronoun,
        },
        "CONTEXT": {
            "DECOY":{
                "dim_1_value": 6, 
                "dim_2_value": 4,
                "pronoun": context_pronoun,
                },
            "PHANTOM_DECOY":{
                "dim_1_value": 8, 
                "dim_2_value": 6,
                "pronoun": context_pronoun,
                "phantom": True,
                },
            }
        }


    alt_defs_categorical_numerical = lambda target_pronoun,competitor_pronoun,context_pronoun: {
        "TARGET": {
            "dim_1_value": educational_degrees_ordered[1],
            "dim_2_value": 4,
            "pronoun": target_pronoun,
            },
        "COMPETITOR": {
            "dim_1_value": educational_degrees_ordered[2], 
            "dim_2_value": 2,
            "pronoun": competitor_pronoun,
        },
        "CONTEXT": {
            "DECOY":{
                "dim_1_value": educational_degrees_ordered[0], 
                "dim_2_value": 3,
                "pronoun": context_pronoun,
                },
            "PHANTOM_DECOY":{
                "dim_1_value": educational_degrees_ordered[2],
                "dim_2_value": 5,
                "pronoun": context_pronoun,
                "phantom": True,
                },
            }
        }
    
    ## CONTEXT SPACE EXPLORATION
    # numerical vs numerical
    dim_1_range=(2,10)
    dim_2_range=(2,10)

    dim_1_values = dim_values(dim_1_range)
    dim_2_values = dim_values(dim_2_range)

    alt_defs_context_grid_numerical_numerical = generate_context_grid(
        target_competitor=alt_defs_numerical_numerical(target_pronoun="",competitor_pronoun="",context_pronoun=""),
        dim_1_values=dim_1_values, 
        dim_2_values=dim_2_values,
        )

    print(alt_defs_context_grid_numerical_numerical("his", "her", "its"))

    # categorical vs numerical
    dim_2_range=(2,10)

    dim_1_values = educational_degrees_ordered
    dim_2_values = dim_values(dim_2_range)

    alt_defs_context_grid_categorical_numerical = generate_context_grid(
        target_competitor=alt_defs_categorical_numerical(target_pronoun="",competitor_pronoun="",context_pronoun=""),
        dim_1_values=dim_1_values, 
        dim_2_values=dim_2_values,
        )

    print(alt_defs_context_grid_categorical_numerical("his", "her", "its"))
    
    # gender fixing
    alt_defs = alt_defs_numerical_numerical(
        target_pronoun="his",
        competitor_pronoun="his",
        context_pronoun="his"
    )

    from prompt_library import job_prompt_assembler, jobs_openings, candidate_phantom_state, job_phantom_requirement

    # Job prompt and candidate templates assembly
    pr_templ, cand_templ = job_prompt_assembler(
        recruiter_instruction="mid_length_1",
        decoy_explanation=0,
        job_opening=jobs_openings["Welder"],
        )

    exp_1 = ContextExperiment(
        target=alt_defs["TARGET"], 
        competitor=alt_defs["COMPETITOR"], 
        context=alt_defs["CONTEXT"], 
        prompt_template=pr_templ,
        candidate_template=cand_templ,
        candidate_phantom_state=candidate_phantom_state,
        job_phantom_requirement=job_phantom_requirement,
        tag="text_experiment_num",
        description="This is a test experiment."
    )

    ###
    alt_defs = alt_defs_context_grid_categorical_numerical(
        target_pronoun="her",
        competitor_pronoun="her",
        context_pronoun="her"
    )

    pr_templ, cand_templ = job_prompt_assembler(
        recruiter_instruction="mid_length_1",
        decoy_explanation=0,
        job_opening=jobs_openings["Mechanical engineer"],
    )

    exp_2 = ContextExperiment(
        target=alt_defs["TARGET"], 
        competitor=alt_defs["COMPETITOR"], 
        context=alt_defs["CONTEXT"], 
        prompt_template=pr_templ,
        candidate_template=cand_templ,
        candidate_phantom_state=candidate_phantom_state,
        job_phantom_requirement=job_phantom_requirement,
        tag="text_experiment_cat",
        description="This is a test experiment."
    )
        
    print(exp_1.run(num_samples=1, metadata={"a": 1, "b": 2}))
    print(exp_2.run(num_samples=1, metadata={"a": 1, "b": 2}))