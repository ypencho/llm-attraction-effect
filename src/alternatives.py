from typing import List
from collections import Counter
from enum import Enum
import itertools as it


class Alternatives:
    """Defines surfaceable choice alternatives and links them to underlying classes.
    """
    def __init__(self, 
                 name_vocabulary: List[str] = ["A", "B", "C"], 
                 treatment_vocabulary: List[str] = ["TARGET", "COMPETITOR", "CONTEXT"],
                 control_vocabulary: List[str] = ["TARGET", "COMPETITOR"]) -> None:
        if len(name_vocabulary) != len(treatment_vocabulary) or not set(control_vocabulary).issubset(treatment_vocabulary):
            raise ValueError("No correspondence between vocabularies!")
        
        self.name_vocabulary = name_vocabulary
        self.treatment_vocabulary = treatment_vocabulary
        self.control_vocabulary = control_vocabulary
    
    def get_enums(self, permutations=False):
        """Enums for alternative names. 
        
        - Dynamic definition of the Enum since alternative names can be changed.
        - Name-alternative type pairs can be permuted to remove the sequence confounding effects.
        
        Returns: a list of dicts with enums for each permutation of alternatives.
        """
        if permutations:
            treatment_vocabulary_list = it.permutations(self.treatment_vocabulary)
        else:
            treatment_vocabulary_list = [self.treatment_vocabulary]
        
        enums = []
        
        for treatment_vocabulary in treatment_vocabulary_list:
            treatment_type_name_dict = dict(zip(treatment_vocabulary, self.name_vocabulary))
            
            control_type_name_dict = {
                t:n 
                for t,n in treatment_type_name_dict.items() 
                if t in self.control_vocabulary
                }

            ControlEnum = Enum(
                'ControlEnum', 
                control_type_name_dict)

            TreatmentEnum = Enum(
                'TreatmentEnum', 
                treatment_type_name_dict)
            
            curr_enum = {
                'control_enum': ControlEnum, 
                'treatment_enum': TreatmentEnum,
                'name_vocabulary': self.name_vocabulary,
            }
            
            enums += [curr_enum]
        return enums
    
    @staticmethod
    def get_enum_name_freq(list_enums): 
        return dict(Counter([v.name for v in list_enums]))
    
    
if __name__ == "__main__":
    dictvaluelist = lambda x: {k:list(v) for k,v in x.items()}
    
    alts = Alternatives()
    
    no_perm = alts.get_enums(permutations=False)
    
    print(dictvaluelist(no_perm[0]))
    
    perm = alts.get_enums(permutations=True)
    
    print(len(perm), [dictvaluelist(p) for p in perm])