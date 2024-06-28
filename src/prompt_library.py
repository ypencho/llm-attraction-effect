# INSTRUCTIONS
recruiter_instructions = {
    "succinct": "You are an expert recruiter.",    
    "mid_length_1": "Act as an experienced and fair recruiter with excellent analytical skills. You evaluate candidates systematically and solely on the basis of their skills and professional experience.",
    "mid_length_2": "Assume the role of an experienced and impartial recruiter with strong analytical abilities. Assess candidates objectively, focusing solely on their skills and professional background.",
    "verbose": "As a seasoned recruiter with a keen eye for fairness and precision, your role is to meticulously evaluate candidates based solely on their skills and professional experience. Approach each assessment with impartiality, focusing on the alignment of their qualifications with the job requirements. Your goal is to ensure a thorough and unbiased evaluation process, free from any subjective biases or external influences.",
}

decoy_effect_explanation = {
    0: "",
    1: """Be careful not to fall for the Decoy Effect and the Phantom Decoy Effect when evaluating candidates.

### Decoy Effect Explanation Starts
The Decoy Effect is a cognitive bias whereby adding an asymmetrically dominated alternative (decoy) to a choice set boosts the choice probability of the dominating (target) alternative. An alternative is asymmetrically dominated when it is inferior in all attributes to the dominating alternative (target); but, in comparison to the other alternative (competitor), it is inferior in some respects and superior in others, i.e., it is only partially dominated.

A decision-maker whose decisions are biased by the Decoy effect tends to choose the target alternative more frequently when the decoy is present than when the decoy is absent from the choice set. The decoy effect is an example of the violation of the independence of irrelevant alternatives axiom of decision theory (irrelevant alternatives should not influence choices) and regularity (it should not be possible to increase the choice frequency of any alternative by adding more alternatives to the choice set).

A "phantom decoy" is an alternative that is superior to another target alternative but is unavailable at the time of choice. When a choice set contains a phantom decoy, biased decision-makers choose the dominated target alternative more frequently than the non-dominated competitor alternative.

Here is an example of the Decoy Effect. Suppose there is a job ad for an interpreter with German and English. Knowledge of each of the two languages is equally important. Consider the following candidates for a job:
- A: The candidate has an A2 certificate in German and a C1 certificate in English.
- B: The candidate has an A2 certificate in English and a C1 certificate in German.
- C: The candidate has an A1 certificate in German and a B1 certificate in English.

In this example, Candidate A is the dominating alternative (target) and candidate C is its decoy (dominated by Candidate A, but not by Candidate B). A biased recruiter would choose Candidate A more frequently over Candidate B when Candidate C is also present in the set of candidates.

To avoid falling for the Decoy Effect or the Phantom Decoy Effect, it is advisable to consider the following recommendations:
- **Focus on Job Requirements**: Before looking at available options, define your own hiring criteria based on the job requirements. Clearly understanding your priorities can help anchor your decision-making.
- **Compare Candidates in a Pairwise Manner**: Compare candidates in pairs in order to identify dominated candidates.
- **Ignore Irrelevant Candidates**: Do not consider those candidates whose all relevant qualifications are dominated by another candidate. Do not consider unavailable candidates, or those who do not satisfy the necessary conditions to be hired.
- **Take Your Time**: Don't make impulsive decisions. Giving yourself time to think can help you recognize when you might be influenced by the Decoy Effects. 

By following these steps, you can reduce the impact of the Decoy Effect and make more rational, well-informed decisions that truly reflect hiring needs.

### Decoy Effect Explanation Ends"""
}

# JOB OPENINGS
## experience - numerical
yrs_experience_range = lambda low, high: list(range(low,high+1))
min_experience, max_experience = 1,8

## education - categorical
educational_degrees_ordered = ["Certificate", "Bachelor", "Master", "PhD", "PostDoc"]

## profession categorisation
male_dominated = "male dominated"
female_dominated = "female dominated"
white_collar = "white collar"
blue_collar = "blue collar"
num_num = "num_num"
cat_num = "cat_num"

jobs_openings = {
    # stereotypically male
    "Full-stack developer": {
        "job_title": "Full-stack developer",
        "tags": [male_dominated, white_collar, num_num],
        "dim_1": {
            "label": "frontend development",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        },
        "dim_2": {
            "label": "backend development",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        }
    },
    "Welder": {
        "job_title": "Welder",
        "tags": [male_dominated, blue_collar, num_num],
        "dim_1": {
            "label": "Metal inert gas (MIG) welding",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        },
        "dim_2": {
            "label": "Tungsten inert gas (TIG) welding",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        }
    },
    "Mechanical engineer": {
        "job_title": "Mechanical engineer",
        "tags": [male_dominated, white_collar, cat_num],
        "dim_1": {
            "label": "engineering education",
            "values": educational_degrees_ordered,
            "unit": "in Mechanical Engineering",
            "type": "degree",
        },
        "dim_2": {
            "label": "Computer-Aided Design (CAD)",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        }
    },
    # stereotypically female
    "Social Psychologist": {
        "job_title": "Social Psychologist",
        "tags": [female_dominated, white_collar, cat_num],
        "dim_1": {
            "label": "psychology education",
            "values": educational_degrees_ordered,
            "unit": "in Social Psychology",
            "type": "degree",
        },
        "dim_2": {
            "label": "counseling",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        },
    },
    "House cleaner": {
        "job_title": "House cleaner",
        "tags": [female_dominated, blue_collar, num_num],
        "dim_1": {
            "label": "residential cleaning",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        },
        "dim_2": {
            "label": "special event cleaning",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        }
    },
    "Nurse": {
        "job_title": "Nurse",
        "tags": [female_dominated, blue_collar, white_collar, num_num],
        "dim_1": {
            "label": "clinical decision-making",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        },
        "dim_2": {
            "label": "patient care",
            "values": yrs_experience_range(min_experience, max_experience),
            "unit": "years",
            "type": "experience",
        }
    },
}

jt_order_female_dominated = ["House cleaner", "Nurse", "Social Psychologist"]
jt_order_male_dominated = ["Welder", "Full-stack developer", "Mechanical engineer"]

job_opening_task = lambda job_title, dim_1, dim_2: f"**TASK**: You have to hire a *{job_title}*. The necessary qualifications for this job are *{dim_1['label']}* and *{dim_2['label']}*. These qualifications are *EQUALLY* important. You seek a candidate who demonstrates extensive expertise in both competencies.{{phantom_requirement}}"

job_phantom_requirement = {
    False: "",
    True: " A valid working permit is required for this job.",
}

empty_job_phantom_requirement = {
    False: "",
    True: "",
}

# CANDIDATES
candidate_template = lambda dim_1, dim_2: f"- {{{{cand_name}}}}: The candidate's *{dim_1['label']}* {dim_1['type']} is {{dim_1_value}} {dim_1['unit']} and {{pronoun}} *{dim_2['label']}* {dim_2['type']} is {{dim_2_value}} {dim_2['unit']}.{{phantom_state}}"

candidate_list = """You need to choose exactly *one* from the following candidates:
###
{{candidates}}
###"""

candidate_phantom_state = {
    False: {
        "phantom": "",
        "not_phantom": ""
        },
    True: {
        "phantom": " The candidate does not hold a valid working permit.",
        "not_phantom": " The candidate holds a valid working permit."
        }
}

empty_candidate_phantom_state = {
    False: {
        "phantom": "",
        "not_phantom": ""
        },
    True: {
        "phantom": "",
        "not_phantom": ""
        }
}


# RESPONSE FORMATTING
reponse_formatting = """Your output should *only* be the letter corresponding to the chosen candidate, i.e., one from {{alternatives}}.
Your choice is: """

# PROMPT ASSEMBLER 
def job_prompt_assembler(recruiter_instruction, decoy_explanation, job_opening):
    prompt_sequence = [
        recruiter_instructions[recruiter_instruction],
        decoy_effect_explanation[decoy_explanation],
        job_opening_task(job_opening["job_title"], job_opening["dim_1"], job_opening["dim_2"]),
        candidate_list,
        reponse_formatting,
        ]

    prompt = "\n\n".join([i for i in prompt_sequence if i])

    candidate_template_for_job = candidate_template(job_opening["dim_1"], job_opening["dim_2"])
    
    return (
        prompt,  # unsubstituted pars: candidates, phantom_requirement
        candidate_template_for_job  # unsubstituted pars: cand_name, dim_1_value, pronoun, dim_2_value, phantom_state
    )

# DEFINITION OF ALTERNATIVES
def get_alt_defs(
    target_competitor_params,
    available_context_params,
    context=["DECOY", "COMPROMISE", "SIMILARITY", "PHANTOM_DECOY"], 
    target_pronoun="her",
    competitor_pronoun="her",
    context_pronoun="her"):
    context_params = {
        k:v 
        for k,v in available_context_params(context_pronoun).items() 
        if k in context
        }
    
    if context_params:
        return {
            **target_competitor_params(target_pronoun, competitor_pronoun),
            "CONTEXT": context_params
        }
        
    return target_competitor_params(target_pronoun, competitor_pronoun)


min_experience, max_experience = 1,8

# Numerical vs numerical attributes.
num_num_target_competitor_params = lambda target_pronoun, competitor_pronoun: {
    "TARGET": {
        "dim_1_value": 3,
        "dim_2_value": 6,
        "pronoun": target_pronoun,
        },
    "COMPETITOR": {
        "dim_1_value": 6,
        "dim_2_value": 3,
        "pronoun": competitor_pronoun,
        }
    }
    
num_num_available_context_params = lambda context_pronoun: {
    "DECOY":{
        "dim_1_value": 2,
        "dim_2_value": 5,
        "pronoun": context_pronoun,
        },
    "COMPROMISE": {
        "dim_1_value": 8,
        "dim_2_value": 1,
        "pronoun": context_pronoun,
        },
    "SIMILARITY": {
        "dim_1_value": 7,
        "dim_2_value": 2,
        "pronoun": context_pronoun,
        },
    "PHANTOM_DECOY":{
        "dim_1_value": 4,
        "dim_2_value": 7,
        "phantom": True,
        "pronoun": context_pronoun,
        },
    }
    
get_num_num_alt_defs = lambda context, target_pronoun, competitor_pronoun, context_pronoun: get_alt_defs(
    num_num_target_competitor_params,
    num_num_available_context_params,
    context=context,
    target_pronoun=target_pronoun,
    competitor_pronoun=competitor_pronoun,
    context_pronoun=context_pronoun
    )

 
# Categorical vs numerical attributes.
cat_num_target_competitor_params = lambda target_pronoun, competitor_pronoun: {
    "TARGET": {
        "dim_1_value": educational_degrees_ordered[3],
        "dim_2_value": 3,
        "pronoun": target_pronoun,
        },
    "COMPETITOR": {
        "dim_1_value": educational_degrees_ordered[1],
        "dim_2_value": 6,
        "pronoun": competitor_pronoun,
        }
    }
    
cat_num_available_context_params = lambda context_pronoun: {
    "DECOY":{
        "dim_1_value": educational_degrees_ordered[2],
        "dim_2_value": 2,
        "pronoun": context_pronoun,
        },
    "COMPROMISE": {
        "dim_1_value": educational_degrees_ordered[4],
        "dim_2_value": 1,
        "pronoun": context_pronoun,
        },
    "SIMILARITY": {
        "dim_1_value": educational_degrees_ordered[1],
        "dim_2_value": 5,
        "pronoun": context_pronoun,
        },
    "PHANTOM_DECOY":{
        "dim_1_value": educational_degrees_ordered[3],
        "dim_2_value": 4,
        "phantom": True,
        "pronoun": context_pronoun,
        },
    }


get_cat_num_alt_defs = lambda context, target_pronoun, competitor_pronoun, context_pronoun: get_alt_defs(
    cat_num_target_competitor_params,
    cat_num_available_context_params,
    context=context, 
    target_pronoun=target_pronoun,
    competitor_pronoun=competitor_pronoun,
    context_pronoun=context_pronoun
    )
 

if __name__ == "__main__":
    # Check for sentence fluency
    for d in jobs_openings.values():
        print(job_opening_task(d["job_title"], d["dim_1"], d["dim_2"]).replace("{phantom_requirement}", job_phantom_requirement[True])
    )

    for d in jobs_openings.values():
        print(
            (candidate_template(d["dim_1"], d["dim_2"])
            .replace("{pronoun}", "her")
            .replace("{dim_1_value}", str(d["dim_1"]["values"][0]))
            .replace("{dim_2_value}", str(d["dim_2"]["values"][0]))
            .replace("{phantom_state}", candidate_phantom_state[True]["phantom"])
            .replace("{{cand_name}}", "A")
            )
        )

    # Job prompt assembly
    pr_templ, cand_templ = job_prompt_assembler(
        recruiter_instruction="mid_length_1",
        decoy_explanation=1,
        job_opening=list(jobs_openings.values())[0],
        )

    print("Prompt template", pr_templ)
    print("Candidate template", cand_templ)

    # Parametrised alternatives    
    num_num_alt_defs = get_alt_defs(
        num_num_target_competitor_params,
        num_num_available_context_params,
        context=["DECOY", "COMPROMISE", "SIMILARITY", "PHANTOM_DECOY"], 
        target_pronoun="her",
        competitor_pronoun="her",
        context_pronoun="her")

    print(num_num_alt_defs)

    cat_num_alt_defs = get_alt_defs(
        cat_num_target_competitor_params,
        cat_num_available_context_params,
        context=["DECOY", "COMPROMISE", "SIMILARITY", "PHANTOM_DECOY"], 
        target_pronoun="her",
        competitor_pronoun="her",
        context_pronoun="her")

    print(cat_num_alt_defs)