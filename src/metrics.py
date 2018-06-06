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
        ref: List of all ground truth equations
        hypo: List of all predicted equations 
        method: Select score metric
        (Important Note): Equations should be given as a list of tokens)

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
    """
    
    Params: 
    references: List of all ground truth equations
    hypotheses: List of corresponding predictions for the equations
    (Note) these equations can be a list of tokens or a string

    Output:
    1 is perfect. Gives 1 - (edit distance)/(max (len(ref), len(hypo))
    """
    edit_dist_tot, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        edit_dist_tot += edit_distance_helper(ref, hypo, len(ref), len(hypo))
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - edit_dist_tot / len_tot
    
def edit_distance_helper(str1, str2, m, n):
    """
    Computes edit distance between two strings or list of tokens 
    Uses DP approach. (Code taken from GeekstoGeeks)
    
    Params:
    str1, str2: Strings being compared
    m, n: Indices of strings being compared

    Output: 
    Edit distance
    """
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
 
    # Fill d[][] in bottom up manner
    for i in range(m+1):
        for j in range(n+1):
            # If first string is empty, only option is to
            # isnert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j
 
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
 
            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
    return dp[m][n]

def edit_distance_score_old(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Uses the distance package. Because this distance package 
    does not work well in the conda environment, we will instead
    be using the hard-coded implementation of this algorithm

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
        (IMPORTANT NOTE): Equations must be represented as a list of tokens
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
    
