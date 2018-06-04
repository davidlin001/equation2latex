# Title: metrics.py
# Author: David Lin
#
#
# Based off of https://github.com/guillaumegenthial/im2latex/blob/master/model/evaluation/text.py 
# ===================
# This module defines scoring metrics to evaluate predictions

import os
import sys
import numpy as np
import distance
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def score(ref, hypo, method):
    """
    Args:
        ref: List of all ground truth equation given as strings
        hypo: List of all predicted equations given as strings
        method: Select score metric

    Output:
        score over all the strings for given method 
    """
    if method == "edit_distance":
        return edit_distance_score(ref,hypo)
    elif method == "bleu":
        return bleu_score(ref,hypo)
    elif method == "exact":
        return exact_match_score(ref,hypo)
    else:
        print("WARNING: Invalid method")
        return -1 


def edit_distance_score(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Params:
        references: List of ground truth equations
        hypotheses: List of predicted equations
    Returns:
        1 is perfect. Gives the edit distance divided by max (len(ref), len(hypo))
    """
    edit_dist_tot, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        edit_dist_tot += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - edit_dist_tot / len_tot


def bleu_score(references, hypotheses):
    """Computes bleu score.
    Params:
        references: list of ground truth equations
        hypotheses: list of predicted equations
    Returns:
        bleu_score. Gives fraction of n-gram matches for n = 1,2,3,4
    """
    # BE CAREFUL OF CASES where equation is less than length 4
    # [ref] necessary for syntax 
    references = [[ref] for ref in references] 
    bleu_score = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu_score


def exact_match_score(references, hypotheses):
    """Computes fraction of exact matches
    Params:
        references: list of ground truth equations
        hypotheses: list of predicted equations
    Returns:
        fraction of exact matches
    """
    score = 0
    for ref, hypo in zip(references, hypotheses):
        if ref == hypo: score += 1
    score /= len(hypotheses)
    return score
    
