from lzma import MODE_NORMAL
import os
from re import M

CONSTANTS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(CONSTANTS_PATH, '..', '..')

DATA_PATH = os.path.join(ROOT, 'data')
DIALOGSUM_PATH = os.path.join(DATA_PATH, 'dialogsum', 'DialogSum_Data')

STRIPPED_DIALOGSUM_PATH = os.path.join(DATA_PATH, 'dialogsum', 'stripped')
STRIPPED_1_PATH = os.path.join(STRIPPED_DIALOGSUM_PATH, 'utterances_person1.json')
STRIPPED_2_PATH = os.path.join(STRIPPED_DIALOGSUM_PATH, 'utterances_person2.json')

OUT_PATH = os.path.join(ROOT, 'out')


PREPOSITION_DEPENDENT = set([
    'if', 'though', 'before', 'although', 'beside', 'besides', 'despite', 'during',
    'unless', 'until', 'via', 'vs', 'upon', 'unlike', 'like', 'with', 'within', 'without', 'because'
])
NOUN_TAGS = set(['NP', 'NN', 'NNP'])
VERB_TAGS = set(['VP', 'VBP'])