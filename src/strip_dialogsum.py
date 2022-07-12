import os

from utils.dialogsum import edit_dialogsum
from utils.constants import STRIPPED_DIALOGSUM_PATH, STRIPPED_1_PATH, STRIPPED_2_PATH


def preprocess_dialogsum():
    print('Strip utterances')
    utts_1, utts_2 = edit_dialogsum('train', 'dev', 'test')

    os.makedirs(STRIPPED_DIALOGSUM_PATH, exist_ok=True)
    
    print('Save utterances of Person 1...')
    utts_1.to_json(STRIPPED_1_PATH)

    print('Save utterances of Person 2...')
    utts_2.to_json(STRIPPED_2_PATH)

    print('Done.')


if __name__ == '__main__':
    preprocess_dialogsum()
