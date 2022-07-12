import os
import pandas as pd

from summarization.monologue import MonoSummarizer
from utils.constants import STRIPPED_1_PATH, STRIPPED_2_PATH, OUT_PATH


def summarize_utterances():
    print('Initialize summarization model ...')
    summarizer = MonoSummarizer()

    print('Perform summarization.')
    summarize_and_write(summarizer, STRIPPED_1_PATH, 'summ_1.txt')
    summarize_and_write(summarizer, STRIPPED_2_PATH, 'summ_2.txt')


def summarize_and_write(model, path, out_name):
    print('Load data')
    data = pd.read_json(path, lines=True).T[0].tolist()
    summarized = model.predict(data)
    
    fname = os.path.join(OUT_PATH, out_name) 
    with open(fname, 'w') as f:
        for summ in summarized:
            text = summ['summary_text'] + '\n'
            f.write(text)


if __name__ == '__main__':
    summarize_utterances()
