import os
import json


def read_jsonl(path):
    json_arr = []
    with open(path, 'r') as json_file:
        for line in json_file.readlines():
            json_obj = json.loads(line)
            json_arr.append(json_obj)
        
    return json_arr


def extract_and_write(label_dest_path, dialogue_dest_path, n=20):
    text_path = './data/dialogsum/DialogSum_Data/dialogsum.test.jsonl'
    json_arr = read_jsonl(text_path)

    with open(label_dest_path, 'w') as label_file, open(dialogue_dest_path, 'w') as dialogue_file:
        for obj in json_arr[:n]:
            dialogue = obj['dialogue']
            print(dialogue)
            dialogue_file.write(dialogue + '\n')

            label = obj['summary1']
            label_file.write(label + '\n')


if __name__ == '__main__':
    extract_and_write('./labels.txt', './dialogue.txt', n=20)