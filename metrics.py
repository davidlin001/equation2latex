# Title: metrics.py
# Author: David Lin
# 
# ===================
# This module defines scoring metrics to evaluate predictions

import distance
from nltk.translate.bleu_score import sentence_bleu

def score(ref, hypo, method):
    """
    
    Params:
    ref: Reference string
    hypo:
    method:

    Output:
    score
    """
    if method == "edit_distance":
        return edit_distance_score(ref,hypo)
    elif method == "bleu":
        return bleu_score(ref,hypo)
    else:
        print("WARNING: Invalid method")
        return -1 


def edit_distance_score(ref,hypo):
    """
    Params:
    ref: Correct String
    hypo: Hypothesis String
    
    Output:
    Edit/ Levenshtein Distance Score. Give the edit distance 
    divided by max (len(ref), len(hypo))
    """
    # To remove spaces if they are there
    # ref = ref.replace(" ","")
    # hypo = hypo.replace(" ", "")
    percent_error = distance.nlevenshtein(ref,hypo,method=1)
    return (1 - percent_error)
    
def bleu_score(ref, hypo):
    """                                                                                    
    Params:                                                                                 
    ref: Correct String                                                                     
    hypo: Hypothesis String                                                                  
    Note: In order to leverage the bleu score, we treat each "gram" as a character in our math expression

    Output:                                                                                  
    bleu_score: 0 to 1 score. 1 is perfect. Checks for percentage of matches 
    among 1,2,3, and 4-grams, giving equal weight to each. Must be extremely careful
    of cases when the length of the equation is < 4 characters. 
    """
    # Be careful of the case 
    assert len(ref),len(hypo) > 4
    ref_list = list(ref)
    hypo_list = list(hypo)
    # Can tune how many n-grams to consider, right now considering up through 4-grams
    return sentence_bleu([ref_list], hypo_list, weights=(0.25, 0.25, 0.25, 0.25))
    #return sentence_bleu([ref_list], hypo_list, weights=(0.33,0.33,0.33,0))

def exact_match_score(ref, hypo):
    pass 
