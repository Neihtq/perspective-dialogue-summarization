import os
import json
import pickle

def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def pickle_file(file_name, data, target_folder=None):
    if target_folder:
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        
    dest_path = os.path.join(target_folder, file_name + ".pkl")
    with open(dest_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    return data


def remove_name(string):
    tmp, flag = "", 0
    for c in string:
        if c == ':':
            flag = 1
            continue
        if flag:
            tmp += c
    
    return tmp


def split_sentences(data):
    sent = []
    for i in range(0, len(data)):
        if len(data[i]['dialogue'].split('\r\n')) > 1:
            sentences = data[i]['dialogue'].split('\r\n')
        else:
            sentences = data[i]['dialogue'].split('\n')
        sent.append(sentences)
    
    return sent


def log_task_and_read_json(file_path, task):
    file_name = file_path.split('/')[-1].split('.')[0]
    print(f'{task} {file_name} split.')
    data = read_json(file_path)

    return data, file_name
