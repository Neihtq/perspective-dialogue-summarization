'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from snorkel.labeling.model import LabelModel
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
import pandas as pd
import re

#from dataset import load_dialogue

WHY = 0
WHAT = 1
WHERE = 2
WHEN = 3
CONFIRM = 4
ABSTAIN = -1


@labeling_function()
def lf_why_keyword(x):
    matches = ['why ', 'y ?', 'whys ', ' y.', ' y .']
    not_matches = ['why not', 'that\'s why', 'I can see why']
    # return WHY if any(item in x.text.lower() for item in matches) else ABSTAIN
    return WHY if (any(item in x.text.lower() for item in matches) and
                   all(item not in x.text.lower() for item in not_matches))else ABSTAIN


@labeling_function()
def lf_what_keyword(x):
    matches = ['what', 'what\'s up', 'what about', 'how']
    return WHAT if (any(item in x.text.lower() for item in matches) and
                    '?' in x.text.lower())else ABSTAIN


@labeling_function()
def lf_where_keyword(x):
    return WHERE if ("where" in x.text.lower() and
                     '?' in x.text.lower()) else ABSTAIN


@labeling_function()
def lf_when_keyword(x):
    matches = ['when']
    soft_matches = ['what time']
    return WHEN if ((any(item in x.text.lower() for item in matches) and
                    '?' in x.text.lower())
                    or (any(item in x.text.lower() for item in soft_matches))) else ABSTAIN

@labeling_function()
def lf_confirm_keyword(x):
    matches = ['are you', 'do you', 'did you', 'can you', 'could you', 'could u', 'have you', 'will you', 'did anyone',
               'can we', 'can I', 'is she', 'is he', 'has she', 'has he', 'has anyone']
    not_matches = ['where ', 'who ', 'when ', 'why ', 'what ', 'how ']
    return CONFIRM if (any(item in x.text.lower() for item in matches) and
                       all(item not in x.text.lower() for item in not_matches)) else ABSTAIN


def filter_function_dialog(dialogs):
    dialog_text = []
    for dialog in dialogs:
        dialog_text += dialog['function_dialogs']


    data = pd.DataFrame(dialog_text, columns=['text'])
    return data


def dispatch_labels(train_data, dialogs):
    """ Dispatch snorkel labels for each dialog"""
    start_idx = 0
    for dialog in dialogs:
        num_function_dialogs = len(dialog['function_dialogs'])
        end_idx = start_idx + num_function_dialogs
        labels_1 = train_data.label_p1[start_idx:end_idx].tolist()
        labels_2 = train_data.label_p2[start_idx:end_idx].tolist()
        dialog['module_index_1'] = labels_1
        dialog['module_index_2'] = labels_2
        start_idx += num_function_dialogs
    return dialogs


def train_snorkel_model(train_data, train_dialogs, L_train):
    label_model = LabelModel(cardinality=6, verbose=True)
    label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
    
    return label_model


def label_data(data, label_model, L_set):
    data['label_p1'] = label_model.predict(L=L_set, tie_break_policy="abstain")
    data['label_p2'] = label_model.predict(L=L_set, tie_break_policy="abstain")


def get_snorkel_label(train_dialogs, eval_dialogs, test_dialogs):
    # Use Train data to train the label_model
    train_data = filter_function_dialog(train_dialogs)
    lfs = [lf_why_keyword, lf_what_keyword, lf_where_keyword, lf_when_keyword, lf_confirm_keyword]
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(train_data)
    
    # Train a model
    label_model = train_snorkel_model(train_data, train_dialogs, L_train)
    label_data(train_data, label_model, L_train)
    snorkel_labeled_train_dialogs = dispatch_labels(train_data, train_dialogs)
    
    # Label the eval dialogs
    eval_data = filter_function_dialog(eval_dialogs)
    L_eval = applier.apply(eval_data)
    label_data(eval_data, label_model, L_eval)
    snorkel_labeled_eval_dialogs = dispatch_labels(eval_data, eval_dialogs)

    # Label the test dialogs
    test_data = filter_function_dialog(test_dialogs)
    L_test = applier.apply(test_data)
    label_data(test_data, label_model, L_test)
    snorkel_labeled_test_dialogs = dispatch_labels(test_data, test_dialogs)

    return snorkel_labeled_train_dialogs, snorkel_labeled_eval_dialogs, snorkel_labeled_test_dialogs

