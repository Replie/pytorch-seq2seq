#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author talm
Description:

"""
import re


def read_question_answers(source_file, target_file, reverse=False):
    # Read the file and split into lines
    question = open(source_file).read().strip().split('\n')
    answers = open(target_file).read().strip().split('\n')

    # Split every line into pairs and normalize
    # pairs = [[normalize_string(s) for s in question] normalize_string(s) for s in answers]

    pairs = zip([normalize_string(s) for s in question], [normalize_string(s) for s in answers])
    # pairs = zip(question, answers)
    # pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        # output_lang = Lang('eng')
    else:
        pairs = [list((p)) for p in pairs]
        # output_lang = Lang('eng')

    return pairs


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?,])", r" \1", s)
    s = re.sub(r"[^a-zA-Zא-ת.!?'`]+", r" ", s)
    return s
