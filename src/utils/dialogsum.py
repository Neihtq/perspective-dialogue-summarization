import os
import pandas as pd

from utils.constants import DIALOGSUM_PATH

def get_dialogues(path: str):
    dialogsum_test_df = pd.read_json(path, lines=True)

    return dialogsum_test_df['dialogue']


def strip_utterances(person: int, df):
    key_word = f'#Person{person}#'
    mapper = lambda dlg: ''.join([f'{utt}\n' for utt in dlg.split('\n') if key_word in utt])

    if person > 2 or person < 1:
        raise Exception(f"For variable 'person' only 1 or 2 allowed. Given value: {person}")

    return df['dialogue'].map(mapper)
            

def edit_dialogsum(*splits):
    dlgs = []
    for split in splits:
        path = os.path.join(DIALOGSUM_PATH, f'dialogsum.{split}.jsonl')
        dlgs.append(get_dialogues(path))

    dialogsum_df = pd.concat(dlgs).reset_index()
    utts_1 = strip_utterances(1, dialogsum_df)
    utts_2 = strip_utterances(2, dialogsum_df)

    return utts_1, utts_2