'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import json
import time
from benepar.spacy_plugin import BeneparComponent
import spacy
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from tqdm import tqdm
import benepar, nltk

from json_loader_helper import load_json, write_json

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
benepar.download('benepar_en3')
"""
Given clean dialogues, for each functional turn, we extract the key phrases which overlap with summaries.
e.g.
'summary': "Megan needn't buy milk and cereals. They're in the drawer next to the fridge."
'dialog': 'Megan: hm , sure , i can do that. but did you check in the drawer next to the fridge ?'
'key_phrases': ['check in the drawer next to the fridge']
"""
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

stemmer = PorterStemmer()
english_stopwords = list(stopwords.words('english'))


def longest_common_subsequence(text1: list, text2: list):
    n, m = len(text1), len(text2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    path = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            stem_text1 = stemmer.stem(text1[i])
            stem_text2 = stemmer.stem(text2[j])
            if stem_text1 == stem_text2 and stem_text2 not in english_stopwords and stem_text2 not in string.punctuation:
                dp[i + 1][j + 1] = dp[i][j] + 1
                path[i + 1][j + 1] = 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
                if dp[i + 1][j + 1] == dp[i][j + 1]:
                    path[i + 1][j + 1] = 2

    word_indexes = []

    def backtrack(i, j):
        if i == 0 or j == 0:
            return
        if path[i][j] == 1:
            backtrack(i - 1, j - 1)
            word_indexes.append(j - 1)
        elif path[i][j] == 2:
            backtrack(i - 1, j)
        else:
            backtrack(i, j - 1)
    backtrack(n, m)
    return dp[-1][-1], word_indexes


def extract_keyphrases(idx, dialogues, tokenized_summary, key_phrases_key):
    dialogues[idx][key_phrases_key] = []
    for sent in dialogues[idx]['function_dialogs']:
        doc = nlp(sent.split(": ")[-1])
        tokenized_sent = [str(token.text) for token in doc]
        parsed_sents = list(doc.sents)

        phrases = []
        for p_sent in parsed_sents:
            phrases.extend([str(i) for i in list(p_sent._.children)])

        lcs_len, word_indexes = longest_common_subsequence(tokenized_summary, tokenized_sent)
        tokenized_phrases, phrase_res = tokenized_sent, []
        if phrases:
            word2phrase, j = dict(), 0
            for i in range(len(tokenized_sent)):
                if tokenized_sent[i] not in phrases[j]:
                    while j != len(phrases) and tokenized_sent[i] not in phrases[j]:
                        j += 1
                    if j == len(phrases):
                        for k in range(i, len(tokenized_sent)):
                            word2phrase[k] = j - 1
                        break
                
                word2phrase[i] = j

            tokenized_phrases = word2phrase

        if lcs_len:
            if phrases: 
                phrase_set = set([tokenized_phrases[i] for i in word_indexes])
                phrase_res = [phrases[phrase_id] for phrase_id in phrase_set if not(phrases[phrase_id] in english_stopwords or phrases[phrase_id] in string.punctuation)]
            else:
                phrase_res = [tokenized_phrases[i] for i in word_indexes]


        dialogues[idx][key_phrases_key].append(phrase_res)


def process_dialogue(dialogues, person):
    summary_key = f'summary_{person}'
    key_phrases_key = f'key_phrases_{person}' 

    for i in tqdm(range(0, len(dialogues))):
        dialogue = dialogues[i]
        tokenized_summary = [str(token.text) for token in nlp(dialogue[summary_key])]
        extract_keyphrases(i, dialogues, tokenized_summary, key_phrases_key)


def chunk_data(data, total_length, chunksize):
    return (data[pos: pos + chunksize] for pos in range(0, total_length, chunksize))


def process_dialogue_file(dialogues_json_path):
    print('load dialogue', dialogues_json_path)
    dialogues = load_json(dialogues_json_path)
    print('process dialogue')
    process_dialogue(dialogues, 'p1')
    process_dialogue(dialogues, 'p2')
    write_json(dialogues_json_path, dialogues)


if __name__ == "__main__":
    st = time.time()
    data_path = '../../../data/processed/new_dialogsum/new_clean_data/'

    print('begin process')
    for split in ['test', 'eval', 'train']:
        process_dialogue_file(os.path.join(data_path, f"{split}.json"))

    #process_dialogue("../SAMsum/clean_data/test.json")
    #process_dialogue("../SAMsum/clean_data/eval.json")
    #process_dialogue("../SAMsum/clean_data/train.json")
    print(time.time() - st)
