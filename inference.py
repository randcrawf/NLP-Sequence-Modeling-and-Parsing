# inference.py

from models import *
from treedata import *
from utils import *
from collections import Counter
from typing import List

import numpy as np


def decode_bad_tagging_model(model: BadTaggingModel, sentence: List[str]) -> List[str]:
    """
    :param sentence: the sequence of words to tag
    :return: the list of tags, which must match the length of the sentence
    """
    pred_tags = []
    for word in sentence:
        if word in model.words_to_tag_counters:
            pred_tags.append(model.words_to_tag_counters[word].most_common(1)[0][0])
        else:
            pred_tags.append("NN") # unks are often NN
    return labeled_sent_from_words_tags(sentence, pred_tags)

#Gets maximum log probability sum and its index
def get_max_sum(model, num_tags, prev_probs, curr_tag_ind):
    maximum = float("-inf")
    max_ind = -1
    for prev_tag_ind in range(num_tags):
        log_prob_sum = model.score_transition(prev_tag_ind, curr_tag_ind) + prev_probs[prev_tag_ind]
        if log_prob_sum > maximum:
            maximum = log_prob_sum
            max_ind = prev_tag_ind

    return maximum, max_ind

#Gets index of max item in an array
def get_max_ind(arr):
    maximum = float("-inf")
    max_ind = -1
    for i in range(len(arr)):
        if arr[i] > maximum:
            maximum = arr[i]
            max_ind = i

    return max_ind



def viterbi_decode(model: HmmTaggingModel, sentence: List[str]) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """
    
    num_tags = len(model.emission_log_probs)
    #[length of sentence x num tags] array containing the probability at each
    dp_probabilities = [[float("-inf") for i in range(num_tags)] for j in range(len(sentence))]
    previous_max_indices = [[-1 for i in range(num_tags)] for j in range(len(sentence))]

    #Handle initial state
    for i in range(num_tags):
        dp_probabilities[0][i] = model.score_init(i) + model.score_emission(sentence, i, 0)

    #Handle itermediate states
    for i in range(1, len(sentence)):
        for j in range(0, num_tags):
            max_sum, max_prev_ind = get_max_sum(model, num_tags, dp_probabilities[i - 1], j)
            dp_probabilities[i][j] = model.score_emission(sentence, j, i) + max_sum
            previous_max_indices[i][j] = max_prev_ind
    
    # print(previous_max_indices[0])
    pred_tags = [None for i in range(len(sentence))]

    #Becktrace through to find answer
    curr_tag_ind = get_max_ind(dp_probabilities[len(sentence) - 1])
    for i in range(len(pred_tags) - 1, -1, -1):
        pred_tags[i] = TaggedToken(sentence[i], model.tag_indexer.get_object(curr_tag_ind))
        curr_tag_ind = previous_max_indices[i][curr_tag_ind]


    return LabeledSentence(pred_tags)

            
class BeamInfo:
    def __init__(self, word, tag_idx, parent_beam_info):
        self.word = word
        self.tag_idx = tag_idx
        self.parent_beam_info = parent_beam_info


def beam_decode(model: HmmTaggingModel, sentence: List[str], beam_size: int) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :param beam_size: the beam size to use
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """

    num_tags = len(model.emission_log_probs)
    if beam_size > num_tags:
        beam_size = num_tags

    curr_beam = Beam(beam_size)

    #Handle initial state
    for i in range(num_tags):
        curr_beam.add(BeamInfo(sentence[0], i, None), model.score_init(i) + model.score_emission(sentence, i, 0))

    prev_beam = curr_beam
    
    #Handle intermediate states
    for i in range(1, len(sentence)):
        curr_beam = Beam(beam_size)
        prev_elts = prev_beam.get_elts()
        for j in range(beam_size):
            for k in range(num_tags):
                curr_beam.add(BeamInfo(sentence[i], k, prev_elts[j]), prev_beam.scores[j] + model.score_transition(prev_elts[j].tag_idx, k) + model.score_emission(sentence, k, i))
        
        prev_beam = curr_beam
    
    pred_tags = [0 for i in range(len(sentence))]
    
    #Backtrace through to find answer
    curr_beam_info = curr_beam.head()
    for i in range(len(sentence) - 1, -1, -1):
        pred_tags[i] = TaggedToken(sentence[i], model.tag_indexer.get_object(curr_beam_info.tag_idx))
        curr_beam_info = curr_beam_info.parent_beam_info

    return LabeledSentence(pred_tags)
    
            

