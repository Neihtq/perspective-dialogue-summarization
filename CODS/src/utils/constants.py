import os
import torch

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

DATA_PATH_RAW = os.path.join(root, 'data/')
DATA_PATH_PROCESSED = os.path.join(root, 'data', 'processed')
NEW_DIALOGSUM_PATH = os.path.join(DATA_PATH_RAW, 'new_dialogsum/')
MINI_DIALOGSUM_PATH = os.path.join(DATA_PATH_PROCESSED, 'new_dialogsum_mini')
TEST_DIALOGSUM_PATH = os.path.join(DATA_PATH_PROCESSED, 'new_dialogsum_clean_data', 'test.json')
SEGMENTATION_PATH = os.path.join(root, 'models', 'save', 'train_segment_predictor')
DIALOGSUM_SEGMENTATION_PATH = os.path.join(SEGMENTATION_PATH, 'pred_test_new.json')

MINI_DIALOGSUM_SEGMENTATION_PATH = os.path.join(root, 'models', 'save', 'mini_dialogsum_TEST', 'pred_test_new.json')


CPU = torch.device('cpu')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUM_TOKEN = "TLDR"
BOS_TOKEN = "<s>"
HL_TOKEN = "<hl>"