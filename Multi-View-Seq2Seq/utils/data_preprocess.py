import os
import re
import torch
import numpy as np

from hmmlearn import hmm
from tqdm import tqdm

from .C99 import C99
from .helper import read_json, pickle_file, load_pickle, split_sentences, log_task_and_read_json
from .constants import (
    SENTENCE_EMBEDDING_PATH,
    SENTENCE_EMBEDDING_SUFFIX,
    TOPIC_SEGMENTATION_PATH,
    TOPIC_SEGMENTATION_SUFFIX,
    STAGE_SEGMENTATION_PATH,
    STAGE_SEGMENTATION_SUFFIX,
    LABELS_PATH
)


def embed_sentences(file_path, embedder):
    data, file_name = log_task_and_read_json(file_path, 'Embedding')
    sent = split_sentences(data)
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sent))):
            embedding = embedder.encode(sent[i])
            embeddings.append(embedding)
    
    file_name = file_name + SENTENCE_EMBEDDING_SUFFIX
    pickle_file(file_name, embeddings, SENTENCE_EMBEDDING_PATH)


def segment_topic(file_path):
    _, file_name = log_task_and_read_json(file_path, 'Topic Segmenting')
    
    sent_embeddings_path = os.path.join(SENTENCE_EMBEDDING_PATH, file_name + SENTENCE_EMBEDDING_SUFFIX)
    sent_embeddings = load_pickle(sent_embeddings_path)

    model = C99(window=4, std_coeff=1)
    sent_label = []
    for i in tqdm(range(len(sent_embeddings))):
        boundary = model.segment(sent_embeddings[i])
        tmp_labels = []
        l = 0
        for j in range(len(boundary)):
            if boundary[j] == 1:
                l += 1
            tmp_labels.append(l)
        sent_label.append(tmp_labels)
    
    file_name = file_name + TOPIC_SEGMENTATION_SUFFIX
    pickle_file(file_name, sent_label, TOPIC_SEGMENTATION_PATH)


def segment_stages(paths):
    sent_embeddings_total = {}
    length, X = [], []
    for path in paths:
        _, file_name = log_task_and_read_json(path, 'Stage Segmenting')
        sent_embeddings_path = os.path.join(SENTENCE_EMBEDDING_PATH, file_name + SENTENCE_EMBEDDING_SUFFIX)
        sent_embeddings = load_pickle(sent_embeddings_path)
        sent_embeddings_total[file_name] = sent_embeddings

        for i in tqdm(range(len(sent_embeddings))):
            length.append(len(sent_embeddings[i]))
            for j in range(len(sent_embeddings[i])):
                X.append(sent_embeddings[i][j])


    remodel = hmm.GaussianHMM(n_components=4, n_iter = 50, covariance_type = 'diag', verbose = True, init_params="cm", params="cmts")
    remodel.startprob_ = np.array([1, 0.0, 0.0, 0.0])
    remodel.transmat_ = np.array([[0.33, 0.34, 0.33, 0],
                                [0.0, 0.33, 0.34, 0.33],
                                [0.0, 0.0, 0.5, 0.5],
                                [0.0, 0.0, 0.0, 1.0]])

    print('Start fitting Gaussian HMM model.')
    remodel.fit(X, length)

    for key, embeddings in sent_embeddings_total.items():
        sent_label = []
        for i in tqdm(range(len(embeddings))):
            labels = remodel.decode(np.array(embeddings[i]))[1]
            sent_label.append(labels)
        

        file_name = key + STAGE_SEGMENTATION_SUFFIX 
        pickle_file(file_name, sent_label, STAGE_SEGMENTATION_PATH)


def concat_conversation(data, labels, sep=' | ', label_type='_sent_c99_label'):
    conversations, summaries_p1, summaries_p2 = [], [], []
    for i in tqdm(range(len(data))):
        if len(data[i]['dialogue'].split('\r\n')) > 1:
            sentences = data[i]['dialogue'].replace(" |", " ").split('\r\n')
        else:
            sentences = data[i]['dialogue'].replace(" |", " ").split('\n')
            
        if len(sentences) == 1:
            continue
         
        summaries_p1.append(data[i]['summary_p1'].strip('\n').replace('\r\nt', ' '))
        summaries_p2.append(data[i]['summary_p2'].strip('\n').replace('\r\nt', ' '))

        if len(labels) > 1:
            tmp = sentences[0]
            for j in range(1, len(sentences)):
                if labels[i][j] != labels[i][j-1]:
                    tmp = tmp + sep + sentences[j]
                else:
                    tmp = tmp + ' ' + sentences[j]

            if label_type == '_sent_c99_label':
                tmp += ' | '
            else:
                tmp = ' | ' + tmp
            conversations.append(tmp)
        elif labels[0] == 1:
            conversations.append(' | ' + ' | '.join(sentences))
        elif labels[0] == 0:
            conversations.append(' | ' + ' '.join(sentences))
        
    return conversations, summaries_p1, summaries_p2


def transform_format(file_path, label_type='_sent_c99_label'):
    data, file_name = log_task_and_read_json(file_path, 'Label Reading')
    folders = {
        TOPIC_SEGMENTATION_SUFFIX: TOPIC_SEGMENTATION_PATH,
        STAGE_SEGMENTATION_SUFFIX: STAGE_SEGMENTATION_PATH
    }

    if label_type != '_all' and label_type != '_none':
        folder = folders[label_type] 
        labels = load_pickle(os.path.join(folder, file_name + label_type + '.pkl'))
    elif label_type == '_all':
        labels = [1]
    elif label_type == '_none':
        labels = [0]
        
    cons, summs_p1, summs_p2 = concat_conversation(data, labels, label_type)

    if not os.path.exists(LABELS_PATH):
        os.mkdir(LABELS_PATH) 

    with open(os.path.join(LABELS_PATH, file_name + label_type +'.source'), 'wt', encoding='utf-8') as source_file:
        for i in range(len(cons)):
            article = cons[i]
            source_file.write(article + '\n')

    with open(os.path.join(LABELS_PATH, file_name + label_type + '_p1.target'), 'wt', encoding='utf-8') as target_file_p1, open(os.path.join(LABELS_PATH, file_name + label_type + '_p2.target'), 'wt', encoding='utf-8') as target_file_p2:
        for i in range(len(summs_p1)):
            summ_p1, summ_p2 = remove_linebreaks(summs_p1[i]), remove_linebreaks(summs_p2[i])
            target_file_p1.write(summ_p1 + '\n')
            target_file_p2.write(summ_p2 + '\n')
            

def remove_linebreaks(summ):
    if '\n' in summ:
        summ = ' '.join(summ.split('\n'))
    
    return summ