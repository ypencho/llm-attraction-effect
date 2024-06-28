import os
import re
import math
from enum import Enum
from tqdm import tqdm
from collections import Counter

from openai import AzureOpenAI
from langchain.output_parsers.enum import EnumOutputParser
from langchain.schema import OutputParserException
from dotenv import load_dotenv

load_dotenv()


class ModelDeployments(Enum):
    """Holds model deplyment names for gpt3.5 and gpt4. 
    """
    GPT35 = 'gpt-35-turbo-instruct'
    GPT4 = 'gpt4'

model_display_names = {
     ModelDeployments.GPT35.value: 'GPT-3.5',
     ModelDeployments.GPT4.value: 'GPT-4',
     }


def get_clients():
    """Initialises two clients - one for each model since models are deployed at different endpoints.
    
    Returns: 
        A dictionary of model deployments and Azure OpenAI clients.
    """
    client_gpt4 = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY_GPT4"),  
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE_GPT4")
    )

    client_gpt35 = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY_GPT35"),  
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE_GPT35")
    )

    return {
        ModelDeployments.GPT35: client_gpt35,
        ModelDeployments.GPT4: client_gpt4,
        }

regex_non_letter_characters = re.compile('[^a-zA-Z]')

def is_string_only_non_letters(string):
    pattern = r'^[^A-Za-z]+$'
    match = re.search(pattern, string)
    
    if match:
        return True
    else:
        return False
    
def get_alternatives_logprobs(top_logprobs, alternatives):
    """Account partially (using available logprobs) for choices' surface form competition by taking into account all tokens which can be matched to the alternatives at hand.

    Args:
        top_logprobs (Dict[str,float]): dictionary of tokens and their logprobs
        alternatives (List[str]): a list of alternatives of interest

    Returns:
        Dict[str,float]: a dictionary with the probability for each alternative.
    """
    alternative_matching_tokens_raw = {
        value:{
            token.replace(value.upper(), "P").replace(value.lower(), "p"):logprob
            for token,logprob in top_logprobs.items()
            if (token in [value.upper(), value.lower()]) or 
            (
                not is_string_only_non_letters(token) and 
                is_string_only_non_letters(token.replace(value.upper(), "").replace(value.lower(), "")) 
            )
        }
        for value in alternatives}

    # alternative variants are matched to each other
    common_alternative_variants = set()

    for _, variants in alternative_matching_tokens_raw.items():
        if not common_alternative_variants:
            common_alternative_variants = set(variants.keys())
        else:
            common_alternative_variants.intersection_update(variants.keys())
    
    alternative_matching_tokens_prob_sum = {
        k:sum([math.exp(logprob)
           for token,logprob in v.items()
           if token in common_alternative_variants])
        for k,v in alternative_matching_tokens_raw.items()
    }
    
    norm_const = sum(alternative_matching_tokens_prob_sum.values())
    
    alternative_matching_tokens_prob_sum_norm = {
        k:v/norm_const
        for k,v in alternative_matching_tokens_prob_sum.items()
        }
       
    return alternative_matching_tokens_prob_sum_norm
    

class LLMRunner:
    """Wraps decision-making methods. Takes a prompt asking to choose between alternatives and returns one chosen alternative.
    """
    def __init__(self, models: ModelDeployments = ModelDeployments) -> None:
        self.models = models
        self.clients = get_clients()
        
    def get_models(self, value=False):
        if value:
            return [e.value for e in self.models]
        
        return self.models

    def choice_logprobs_gpt35(self, prompt, decoding_enum, temperature=1, max_retries=5):
        """Given and prompt asking for a choice, find the probability for each alternative using logprobs. 

        Args:
            prompt (str): a prompt asking to choose between alternatives
            decoding_enum (Enum): an Enum containing the alternatives and their type/class
            temperature (int, optional): _description_. Defaults to 1.

        Returns:
            Dict[str,float]: a dcitionary with probabilities for each decoded alternative.
        """
        logprob_model = self.models.GPT35  # an OpenAI model supporting logprobs
        
        for _ in range(max_retries):
            result = self.clients[logprob_model].completions.create(
                model=logprob_model.value,
                prompt=prompt,
                logprobs=100,  # maximal number of logprobs returned by the service
                temperature=temperature,
                max_tokens=1,  # generate only one token - the choice MUST be represented by a single character! 
            )
            
            try:
                top_logprobs = result.choices[0].logprobs.top_logprobs[0]
                break
            except Exception as e:
                print("Log probs error", e)
                # print(result)
                # print(prompt)
                
        values = [v.value for v in decoding_enum]

        alt_logprobs = get_alternatives_logprobs(top_logprobs, values)
        
        enum_parser = EnumOutputParser(enum=decoding_enum)

        decoded_logprobs = {
            enum_parser.parse(k).name:v 
            for k,v in alt_logprobs.items()
            }
        
        return decoded_logprobs

    def choice_sample_gpt4(self, prompt, decoding_enum=None, temperature=1., num_samples=10, max_iters_per_sample=3, max_tokens=10):
        sampling_model = self.models.GPT4
        
        if decoding_enum:
            enum_parser = EnumOutputParser(enum=decoding_enum)

        choices = []

        max_iters = num_samples*max_iters_per_sample
        counter = 0
        
        with tqdm(total=num_samples, desc='Choice sampling') as pbar:
            while len(choices) < num_samples:
                if counter > max_iters:
                    break
                else:
                    counter += 1
                    
                try:
                    result = self.clients[sampling_model].chat.completions.create(
                        model=sampling_model.value,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                            ]
                        )
                    
                    answer = regex_non_letter_characters.sub("", result.choices[0].message.content)  # remove non-letter characters from the response
                except Exception as e:
                    print("Error in chat completion.", e)
                    continue
                
                try:
                    if decoding_enum:
                        answer_decoded = enum_parser.parse(answer).name
                    else:
                        answer_decoded = answer
                except OutputParserException as e:
                    print("OutputParserException", e)
                    continue
                
                choices += [answer_decoded]
                pbar.update()
        
        choice_counts = dict(Counter(choices))
                
        return choice_counts


if __name__ == "__main__":
    alternatives = ["X","Y","Z"]
    
    top_logprobs = {
        'X': -0.6023186,
        'Y': -1.4267299,
        'Z': -1.8990505,
        '\n': -2.9093826,
        ' X': -5.661702,
        ' Y': -6.1420126,
        ' Z': -6.7434855,
        '\n\n': -8.437005,
        'YZ': -12.34275,
        'z': -12.380927,
        '#': -12.429935,
        'y': -12.460035,
        }
    
    print(get_alternatives_logprobs(top_logprobs, alternatives))
    
    print(get_clients())
    
    llm_runner = LLMRunner()
    
    print(llm_runner.models.GPT35.value)
    print(llm_runner.get_models())
    
    prompt = "Choose the more colourful bird:\nA: Rainbow Lorikeet\nB: Scarlet Macaw\n\n Answer only with A or B, nothing else."
    decoding_enum = Enum('Colorful', {"Psittaculidae": "A","Psittacidae":"B" })
    
    gpt3_choices = llm_runner.choice_logprobs_gpt35(prompt=prompt, decoding_enum=decoding_enum)
    
    print("gpt3_choices", gpt3_choices)
    
    gpt4_choices = llm_runner.choice_sample_gpt4(prompt=prompt, decoding_enum=decoding_enum)
    
    print("gpt4_choices", gpt4_choices)