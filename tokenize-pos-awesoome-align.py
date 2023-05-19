import os
import sys
import json
import pandas as pd
import pickle 
import numpy as np
from collections import Counter


from tqdm import tqdm
tqdm.pandas()


from nltk.tokenize import sent_tokenize
from nltk.tree import Tree
from nltk.draw.tree import TreeView
from nltk.tokenize import sent_tokenize
from conllu import parse

from indicnlp.tokenize import sentence_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


import stanza

import re
import requests
import argparse

import traceback

os.getcwd()


df_translations_set = pd.read_json("unique_utterances_en_hi_transltions.json")
# df_translations_set = df_translations_set[:1]
print(df_translations_set.shape)
# df_translations_set.head()


nlp_en = stanza.Pipeline(lang='en', processors='tokenize, pos')
nlp_hi = stanza.Pipeline(lang = 'hi', processors='tokenize, pos')


sys.path.append("/scratch/anmol.goel/stanford-tod/codemix-gcm/")

from auxilaries.utils import awesomealign


aligner = awesomealign(modelpath = '/scratch/anmol.goel/stanford-tod/codemix-gcm/awesome-align-sbatch/awesome-align-fine-tuned-savedmodel',
                      tokenizerpath = '/scratch/anmol.goel/stanford-tod/codemix-gcm/awesome-align-sbatch/awesome-align-fine-tuned-savedmodel')  


def get_stanza_info(text, model):
    
    doc = model(text)
    
    sents, tokens, postags = [], [], []
    
    for sentence in doc.sentences:
        sents.append(' '.join([f'{token.text}' for token in sentence.tokens]))
        tokens.append([f'{token.text}' for token in sentence.words])
        postags.append([f'{token.upos}' for token in sentence.words])

    return {"sentences" : sents,
           "tokens" : tokens,
           "postags" : postags}


en_tokenized, en_pos, hi_tokenized, hi_pos = [], [], [], []

for ind, row in tqdm(df_translations_set.iterrows()):
    
    eng_feats = get_stanza_info(row["en"], nlp_en)
    hin_feats = get_stanza_info(row["hi"], nlp_hi)
    
    en_tokenized.append(eng_feats["tokens"])
    hi_tokenized.append(hin_feats["tokens"])    
    
    en_pos.append(eng_feats["postags"])
    hi_pos.append(hin_feats["postags"])    

#print(f"en tokens : {eng_feats['tokens']}")

df_translations_set["en_tokens"] = en_tokenized
df_translations_set["en_pos"] = en_pos
df_translations_set["hi_tokens"] = hi_tokenized
df_translations_set["hi_pos"] = hi_pos

def create_alignments_token_map(sent_src, sent_tgt, alignments):
    
    token_map = {}
    sent_src = sent_src.split()
    sent_tgt = sent_tgt.split()
    
    for el in alignments.split():
        el = el.split("-")
        try:
            token_map[sent_src[int(el[0])]] = sent_tgt[int(el[1])]
            token_map[sent_tgt[int(el[1])]] = sent_src[int(el[0])]
        except IndexError:
            print("index error")
            print(sent_src, sent_tgt, alignments)
            print("-"*20)
            token_map = None
    
    return token_map



def get_alingment_token_map(en_sent, hi_sent):
    alignments = aligner.get_alignments_sentence_pair(en_sent, hi_sent)
    token_alignment_map = create_alignments_token_map(en_sent, hi_sent, alignments)
    return alignments, token_alignment_map

list_of_alignments, list_of_token_alignment_map = [], []

for ind, row in tqdm(df_translations_set.iterrows()):
    #print(row["en_tokens"])
    en_tokenized_sent = [" ".join(sent_list) for sent_list in row["en_tokens"]]
    hi_tokenized_sent = [" ".join(sent_list) for sent_list in row["hi_tokens"]]
    alignment_row, token_alignment_map_row = [], []
    try:
        assert len(en_tokenized_sent) == len(hi_tokenized_sent)
        for ensent, hisent in zip(en_tokenized_sent, hi_tokenized_sent):
            alignments, token_alignment_map = get_alingment_token_map(ensent, hisent)
            alignment_row.append(alignments)
            token_alignment_map_row.append(token_alignment_map)
        list_of_alignments.append(alignment_row)
        list_of_token_alignment_map.append(token_alignment_map_row)

    except AssertionError:
        alignment_row, token_alignment_map_row = None, None
        list_of_alignments.append(None)
        list_of_token_alignment_map.append(None)



df_translations_set["alignments_awesomealign"] = list_of_alignments
df_translations_set["token_alignment_map_awesomealign"] = list_of_token_alignment_map


df_translations_set.to_json("train_unique_utterances_en_hi_transltions_token_pos_alignments.json", 
                                          force_ascii = False, 
                                         orient = "records",
                                        indent = 4)

