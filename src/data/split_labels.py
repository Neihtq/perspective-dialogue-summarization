import os
import json
import argparse
import pandas as pd

from nltk import Tree
from nltk.tree.parented import ParentedTree
from tqdm import tqdm
from difflib import SequenceMatcher
from allennlp.predictors.predictor import Predictor

parser = argparse.ArgumentParser(description='Split labels and build new DialogSum dataset.')
parser.add_argument('-path', help='Path to DialogSum dataset', required=True)
args = parser.parse_args()


PREPOSITION_DEPENDENT = set([
    'if', 'though', 'before', 'although', 'beside', 'besides', 'despite', 'during',
    'unless', 'until', 'via', 'vs', 'upon', 'unlike', 'like', 'with', 'within', 'without', 'because'
])
NOUN_TAGS = set(['NP', 'NN', 'NNP'])
VERB_TAGS = set(['VP', 'VBP'])
EOS_TOKEN = '<EOS>'

def get_predictor(model):
    if model == 'pos':
        url = "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
    elif model == 'ner':
        url = "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz"
    
    return Predictor.from_path(url)


def show_dialogue_and_summaries(dialogsum_df, summaries, i):
    for dlg in dialogsum_df.loc[i]['dialogue'].split('\n'):
        print(dlg)

    print('\n')
    for _, (summ, idx) in enumerate(summaries):
        if idx == i:
            print(summ)
            break

def prepare_dataframe(dialogsum_path):
    test_path = os.path.join(dialogsum_path, 'dialogsum.test.jsonl')
    dialogsum_test_df = pd.read_json(test_path, lines=True)
    dialogsum_test_df = dialogsum_test_df.rename(columns={"summary1": "summary"})
    dialogsum_test_df['split'] = 'test'
    
    dev_path = os.path.join(dialogsum_path, 'dialogsum.dev.jsonl')
    dialogsum_dev_df = pd.read_json(dev_path, lines=True)
    dialogsum_dev_df['split'] = 'val'

    train_path = os.path.join(dialogsum_path, 'dialogsum.train.jsonl')
    dialogsum_train_df = pd.read_json(train_path, lines=True)
    dialogsum_train_df['split'] = 'train'
    
    dialogsum_df = pd.concat([dialogsum_train_df, dialogsum_dev_df, dialogsum_test_df])
    dialogsum_df.reset_index(inplace=True)
    
    return dialogsum_df

def get_summs_w_2_persons(dialogsum_df):
    summaries = []
    two_person_dlg_f = dialogsum_df[dialogsum_df['dialogue'].str.contains('#Person3#') == False]
    indices = dialogsum_df.index[dialogsum_df['dialogue'].str.contains('#Person3#') == False].tolist()
    summaries = [(dialogsum_df.loc[idx]['summary'], idx) for idx in indices]
    
    return summaries, two_person_dlg_f

def split_sentence(pred):
    sents = []
    sent = []
    for word, token in zip(pred['tokens'], pred['pos_tags']):
        if token == '.':
            sent[-1] += word
            sents.append(sent)
            sent = []
        else:
            sent.append(word)
            
    return [" ".join(s) for s in sents]

def get_sentences(pred):
    root = pred['hierplane_tree']['root']
    sents = []
    for child in pred['hierplane_tree']['root']['children']:
        if child['nodeType'] == 'S':
            sents.append(child)    
    
    
    return sents if sents else [root]

def split_summaries(summaries, predictor):
    summaries_split = {}
    for summ, i in tqdm(summaries):
        out = predictor.predict(summ)
        summaries_split[i] = split_sentence(out)

    return summaries_split

def get_split_summaries(summ_split_path, summaries, predictor, force_rerun=False):
    if os.path.exists(summ_split_path) and not force_rerun:
        with open(summ_split_path, 'r') as json_file:
            splits = json.load(json_file)
            summaries_split = {int(k): splits[k] for k in splits}
    else:
        summaries_split = split_summaries(summaries, predictor)
        with open(summ_split_path, 'w') as json_file:
            json.dump(summaries_split, json_file)
    
    return summaries_split

def cleanup_summs(summaries_split):
    for k in tqdm(summaries_split):
        for i in range(len(summaries_split[k])):
            summaries_split[k][i] = summaries_split[k][i].replace('#Person1#', 'XYZ1')
            summaries_split[k][i] = summaries_split[k][i].replace('# Person1#', 'XYZ1')
            summaries_split[k][i] = summaries_split[k][i].replace('# Person1 #', 'XYZ1')
            summaries_split[k][i] = summaries_split[k][i].replace('#Person1 #', 'XYZ1')

            summaries_split[k][i] = summaries_split[k][i].replace('#Person2#', 'XYZ2')
            summaries_split[k][i] = summaries_split[k][i].replace('# Person2#', 'XYZ2')
            summaries_split[k][i] = summaries_split[k][i].replace('# Person2 #', 'XYZ2')
            summaries_split[k][i] = summaries_split[k][i].replace('#Person2 #', 'XYZ2')

            summaries_split[k][i] = summaries_split[k][i].replace(" 'll", "'ll")
            summaries_split[k][i] = summaries_split[k][i].replace(" 's", "'s")
            summaries_split[k][i] = summaries_split[k][i].replace(" n't", "n't")
            summaries_split[k][i] = summaries_split[k][i].replace(" - ", "-")
            summaries_split[k][i] = summaries_split[k][i].replace(" ,", ",")
            summaries_split[k][i] = summaries_split[k][i].replace(" 've'", "'ve'")
            summaries_split[k][i] = summaries_split[k][i].replace("..", ".")
            summaries_split[k][i] = summaries_split[k][i].replace('. ', ' ')
            
def tag_summs(summaries_split, predictor):
    tagged_summs = {}
    curr = None

    for k in tqdm(summaries_split):
        curr = k
        split = summaries_split[k]
        tagged_summs[k] = []
        for summ in split:
            ptree = predict_and_get_tree(summ, predictor)
            subs = get_subsentences(ptree)

            clauses = get_clauses(subs)
            tagged_summs[k].extend(clauses)
    
    return tagged_summs

def correct_tags(tagged_summs, summaries_split):
    count = 0
    for k in tagged_summs:
        if len(summaries_split[k]) > len(tagged_summs[k]):
            count += 1
            tagged_summs[k] = summaries_split[k][::]
        
def cleanup_tags(tagged_summs):
    for k in tqdm(tagged_summs):
        i = 0
        while i < len(tagged_summs[k]):
            if tagged_summs[k][i] == '.':
                del tagged_summs[k][i]
            else:
                if isinstance(tagged_summs[k][i], list):
                    tagged_summs[k][i] = ' '.join(tagged_summs[k][i])
                tagged_summs[k][i] = tagged_summs[k][i].replace("# Person1 #", "#Person1#")
                tagged_summs[k][i] = tagged_summs[k][i].replace("# Person2 #", "#Person2#")
                tagged_summs[k][i] = tagged_summs[k][i].replace(" 'll", "'ll")
                tagged_summs[k][i] = tagged_summs[k][i].replace(" 's", "'s")
                tagged_summs[k][i] = tagged_summs[k][i].replace(" ,", ",")
                tagged_summs[k][i] = tagged_summs[k][i].replace(" '", "'")
                tagged_summs[k][i] = tagged_summs[k][i].replace(" - ", "-")
                tagged_summs[k][i] = tagged_summs[k][i].replace(". ", " ")
                if len(tagged_summs[k][i]) > 0:
                    if tagged_summs[k][i][-1] != '.':
                        tagged_summs[k][i] = tagged_summs[k][i] + EOS_TOKEN#'.'
                tagged_summs[k][i] = tagged_summs[k][i].replace("#Person1#", "XYZ1")
                tagged_summs[k][i] = tagged_summs[k][i].replace("#Person2#", "XYZ2")
                i += 1

    # hard coded corrections

    solutions = {
        259: {0 : 'Alice wants to apply for a scholarship offered by the American Minority Students Scholarship Association since she is eligible for it that she is Asian American, a student in junior year and has GPA 3.92.',},
        298: {1: 'They both play bridge'},
        4446: {1: 'XYZ1 thinks working overtime is not always pleasant.'},
        3363: {2: "there'll be more collections of his works."},
        4250: {1: 'Jason comforts her.'},
        8760: {1: "They 've got meat, utensils and paper plates, and are going to buy some buns and ketchup."},
        10203: {2: "they've made a room reservation."},
        11213: {1: "They've been dating for three years."},
        11832: {0: "Say forgets to take Melber's book and suggest they pick it up after the show."}
    }
    for i, idx in [(259, 0), (298, 1), (4446, 1), (4250, 1), (3363, 2), (8760, 1), (10203, 2), (11213, 1), (11832, 0)]:
        try:
            tagged_summs[i][idx] = solutions[i][idx]
        except:
            print(f'Could not resolve summary for {i} and {idx}')
    try:
        tagged_summs[259].pop(1)
    except:
        print("Could not perform pop oin summary 259 no 1")
    
    try:
        tagged_summs[2016] = ["Edward Smith wants to book a flight to New York on July 21st but it isn't available.", "he takes another flight on July 22nd."]
        tagged_summs[744] = ["XYZ1 sends a necklace to Mom on Mother's Day"] + tagged_summs[744]
        tagged_summs[6934] = ['XYZ1 is going to buy bicycle A5, FOB Qingdao from Mr Smith.', 'they agree on 3.5%.']
    except:
        print('Could not correct remaining summaries')
    
            
def get_tagged_summs(tagged_summs_path, summaries_split, predictor, force_rerun=False):
    print(not force_rerun)
    if os.path.exists(tagged_summs_path) and not force_rerun:
        with open(tagged_summs_path, 'r') as json_file:
            json_data = json.load(json_file)
            tagged_summs = {int(k): json_data[k] for k in json_data}
    else:
        tagged_summs = tag_summs(summaries_split, predictor)
        with open(tagged_summs_path, 'w') as json_file:
            json.dump(tagged_summs, json_file)

    
    correct_tags(tagged_summs, summaries_split)
    cleanup_tags(tagged_summs)
    remove_dots_and_empty(tagged_summs)
    
    return tagged_summs


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_clauses(subsentences):
    clauses = []
    for subsent in subsentences:
        clause = []
        i = 0
        while i < len(subsent):
            child = subsent[i]
            if child[0] == 'and':
                clause.extend(child.leaves())
                i += 1
                child = subsent[i]
                clause.extend(child.leaves())
            elif child.label() == 'SBAR':
                if child[0].label() == 'IN' and child[0][0].lower() not in PREPOSITION_DEPENDENT:
                    clauses.append(clause)
                    clause = []
                    clauses.append(child.leaves()[1:])
                else:
                    clause.extend(child.leaves())

            elif child.right_sibling():
                if child.label() == 'PP' and child.right_sibling().label() in NOUN_TAGS:
                    clause.extend(child.leaves())
                    
                if child.label() in NOUN_TAGS:
                    clause.extend(child.leaves())
                    if child.right_sibling().label() != 'SBAR':
                        clause.extend(child.right_sibling().leaves())
                    if child.right_sibling().right_sibling():                        
                        if child.right_sibling().label() == 'ADVP' :
                            clause.extend(child.right_sibling().right_sibling().leaves())
                        if child.right_sibling().label() == 'JJ' and child.right_sibling().right_sibling().label() == 'PP':
                            clause.extend(child.right_sibling().right_sibling().leaves())
                        elif child.right_sibling().right_sibling().label() == 'VP':
                            clause.extend(child.right_sibling().right_sibling().leaves())

            i += 1
        if clause:
            clauses.append(clause)

    return clauses

def get_subsentences(tree):
    subs = []
    tmp = []
    for i, child in enumerate(tree):
        if child.label() != 'CC':
            tmp.append(child)
        if child.label() == 'S':
            if tmp:
                subs.append(tmp)
                tmp = []
            subs.append(child)

    return subs if subs else [tree]

def contains_title(subject):
    lower_string = [string.lower() for string in subject]
    titles = ['mr', 'mrs', 'ms', 'mister', 'miss', 'misses', 'dr', 'doctor']
    for title in titles:
        if title in lower_string:
            return True
        
    return False
    
def get_subject(tree):
    output = []
    for child in tree:
        if child.label() in NOUN_TAGS:
            output = child.leaves()
            break
        elif child.label() == 'S' or child.label() == 'SBAR':
            output = get_subject(child)
            if output:
                break
        elif child.label() == 'VP':
            if child.left_sibling():
                output = child.left_sibling().leaves()
            else:
                output = get_subject(child)
            break
    
    if len(output) >= 2:
        if 'and' in output:
            output = output
        elif contains_title(output):

            output = [output[1]]
        else:
            output = [output[0]]
        
        return output
     
    return output

def predict_and_get_tree(summ, predictor=None):
    if predictor is None:
        predictor = get_predictor('pos')
    pred = predictor.predict(summ)
    t = Tree.fromstring(pred['trees'])
    ptree = ParentedTree.convert(t)

    return ptree

def has_top_level_NP(tree):
    NOUN_TAGS = set(['NP', 'NN', 'NNP'])
    for child in tree:
        if child.label() == 'S' or child.label() == 'SBAR':
            return has_top_level_NP(child)
        if child.label() == 'VP' or child.label() in NOUN_TAGS:
            return True
        
    return False

def get_names(pred):
    names = []
    for word, tag in zip(pred['words'], pred['tags']):
        if 'PER' in tag:
            names.append(word.lower())
    
    return names

def get_all_names(tagged_summs, predictor):
    names = []
    for k in tqdm(tagged_summs):
        for sent in tagged_summs[k]:
            pred = predictor.predict(sent)
            names += get_names(pred)
    
    names_set = set(names)
    names_set.add('xyz1')
    names_set.add('xyz2')
    return names_set

def get_summs_w_they(tagged_summs):
    with_they = []
    for k in tagged_summs:
        for i in range(len(tagged_summs[k])):
            if 'they' in tagged_summs[k][i]:
                with_they.append((k,i))
    
    return with_they

def get_none_and_theyNP(with_they, tagged_summs, predictor):
    they_NP = []
    is_none = []
    for k, i in tqdm(with_they):
        pair = (k, i)
        summ = tagged_summs[k][i]
        pred = predictor.predict(summ)
        t = Tree.fromstring(pred['trees'])
        ptree = ParentedTree.convert(t)
        subject = get_subject(ptree)
        if subject:
            if 'they' in subject.lower():
                they_NP.append(pair)
        else:
            is_none.append(pair)
            
    return they_NP, is_none

def distinct_prepositions(summaries, predictor):
    preps = set()
    for k in tqdm(summaries):
        split = summaries[k]
        for sent in split:
            out = predictor.predict(sentence=sent)
            t = Tree.fromstring(out['trees'])
            ptree = ParentedTree.convert(t)
            for subtree in ptree.subtrees(filter=lambda x: x.label() == 'IN'):
                preps.add(subtree[0])
    return preps

def remove_dots_and_empty(tagged_summs):
    for k in tagged_summs:
        i = 0
        while i < len(tagged_summs[k]):
            if tagged_summs[k][i] == '.' or tagged_summs[k][i] == '':
                del tagged_summs[k][i]
            else:
                i += 1


def assign_exceptions(k, i):
    p_1 = (True, 'Person1')
    p_2 = (True, 'Person2')
    exceptions = {
        (0, 0): p_1,
        (7, 4): p_1,
        (950, 0): p_1,
        (11213, 1): p_2,
        (7233, 1): p_1,
        (11832, 0): p_1,
    }
    is_exception, key = exceptions.get((k, i), (False, None))
    
    return is_exception, key


def assign_labels(tagged_summs, names_set, dialogsum_df, labeled_summs, none_subjects, predictor=None, start_idx=0, end_idx=13460):
    if predictor is None:
        predictor = get_predictor('pos')

    for k in tqdm(range(start_idx, end_idx)):
        if k not in tagged_summs:
            continue
        labeled_summs[k] = {'Person1': [], 'Person2': []}
        prev = None
        for i in range(len(tagged_summs[k])):
            summ = tagged_summs[k][i]
            
            is_exception, key = assign_exceptions(k, i)
            if is_exception:
                labeled_summs[k][key].append(summ)
                continue
            
            tree = predict_and_get_tree(summ, predictor)
            
            subject = get_subject(tree)
            if subject:
                subject = ' '.join(subject).lower()
                for suffix in ["'s", "'ll", "'ve", "'"]:
                    subject = subject.replace(suffix, "")

                if 'and' in subject:
                    target_keys = []
                    subjects = subject.split('and')
                    for subj in subjects:
                        target_keys += define_speaker(subj.strip(), names_set, dialogsum_df, k, prev)
                else:
                    target_keys = define_speaker(subject.lower(), names_set, dialogsum_df, k, prev)
                
                if not target_keys:
                    none_subjects.append((k, i))
                    continue
                
                for key in target_keys:
                    labeled_summs[k][key].append(summ)
            else:
                none_subjects.append((k, i))
            prev = key
                
    return labeled_summs, none_subjects

def define_speaker(subject, names_set, dialogsum_df, idx, prev):
    if subject == 'xyz1':
        keys = ['Person1']
    elif subject == 'xyz2':
        keys = ['Person2']
    elif subject in names_set:
        keys = search_speaker(subject, dialogsum_df.loc[idx]['dialogue'])
    else:
        keys = pronoun_distinction(subject, prev)
    
    return keys

def search_speaker(name, dialogue):
    person = None
    for utt in dialogue.split('\n'):
        if name in utt.lower():
            person = get_speaker(utt)
            break
        else:
            for word in utt.split(' '):
                if similar(word.lower(), name) >= 0.75:
                    person = get_speaker(utt)
                    return [person]
                    
    
    return [person] if person else []


def get_speaker(utt):
    intro_sents = ["i am", "i'm", "name is", "name's", "this is", "that is", "that's"]
    introduces = False
    for sent in intro_sents:
        if sent in utt:
            introduces = True
            break
    speaker = utt.split(' ')[0]

    person = 'Person2' if '1' in speaker else 'Person1'
    if introduces:
        person = 'Person1' if '1' in speaker else 'Person2'

    return person

def pronoun_distinction(pronoun, prev_label):
    singular = set(['he', 'she', 'his', 'her', 'him'])
    if pronoun in singular:
        return [prev_label] if prev_label else []
    
    return ['Person1', 'Person2']


def write_labels_to_json(split, data):
    with open(os.path.join('..', '..','data', 'new_labels', f'new_dialogsum_labels_{split}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)


def create_new_labels():
    folder = '../../data/new_labels'
    if not os.path.exists(folder):
        os.makedirs(folder)

    print('Get predictor')
    predictor = get_predictor('pos')
    print('Load dataframes')
    dialogsum_df = prepare_dataframe(args.path)
    print('Extract samples only with 2 persons')
    summaries, dialogsum_df = get_summs_w_2_persons(dialogsum_df)
    
    summ_split_path = 'summaries_split.json'
    print('Split summaries')
    summaries_split = get_split_summaries(summ_split_path, summaries, predictor, force_rerun=True)
    print('cleanup summaries')
    cleanup_tags(summaries_split)

    print('Tag summaries')
    tagged_summs_path = os.path.join(folder, 'tagged_summs.json')
    tagged_summs = get_tagged_summs(tagged_summs_path, summaries_split, predictor, force_rerun=True)

    print('Get NER predictor')
    predictor = get_predictor('ner')
    print('Get all names')
    names_set = get_all_names(tagged_summs, predictor)

    none_subjects, labeled_summs = [], {}
    predictor = get_predictor('pos')
    print('Assign labels')
    label_assignments, errors = assign_labels(tagged_summs, names_set, dialogsum_df, labeled_summs, none_subjects, predictor, end_idx=len(dialogsum_df))

    print('Write label assignment')
    with open(os.path.join(folder, 'labels.json'), 'w') as json_file:
        json.dump(label_assignments, json_file)

    labels = label_assignments

    print('Reset person tokens')
    for key in labels:
        for person in labels[key]:
            for i in range(len(labels[key][person])):
                labels[key][person][i] = labels[key][person][i].replace('XYZ1', 'Person1')
                labels[key][person][i] = labels[key][person][i].replace('XYZ2', 'Person2')

    print('Gather empty labels')
    empties = []
    for key in labels:
        for person in labels[key]:
            if len(labels[key][person]) == 0:
                empties.append(key)
                labels[key][person].append(dialogsum_df.loc[key]['summary'] + '#TOREMOVE#')


    print('Add EOS Token')
    for key in labels:
        for person in labels[key]:
            if '#TOREMOVE#' in labels[key][person][0]:
                labels[key][person][0] = labels[key][person][0].replace('#TOREMOVE#', '')
            for i in range(len(labels[key][person])):
                if labels[key][person][i][-len(EOS_TOKEN):] != EOS_TOKEN:#'.':
                    labels[key][person][i] += EOS_TOKEN#'.'


    print('Write corrected labels')
    with open(os.path.join(folder, 'labels_corrected.json'), 'w') as json_file:
        json.dump(labels, json_file, indent=4)

    dialogsum_new_labels = {
        'test': {}, 'val': {} , 'train': {}
    }

    print('split labels')
    for key in labels:
        split = dialogsum_df.loc[key]['split']
        dialogsum_new_labels[split][key] = labels[key]
        

    print('Write splits')
    for split in dialogsum_new_labels:
        data = dialogsum_new_labels[split]
        print(split, len(data))
        write_labels_to_json(split, data)


if __name__ == '__main__':
    create_new_labels()