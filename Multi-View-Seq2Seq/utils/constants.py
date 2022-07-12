import os

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

DATA_PATH = os.path.join(root, 'data/')
NEW_DIALOGSUM_PATH = os.path.join(DATA_PATH, 'new_dialogsum/')
SENTENCE_EMBEDDING_PATH = os.path.join(DATA_PATH, 'sentence_embeddings')
TOPIC_SEGMENTATION_PATH = os.path.join(DATA_PATH, 'topic_segmentations')
STAGE_SEGMENTATION_PATH = os.path.join(DATA_PATH, 'stage_segmentations')
LABELS_PATH = os.path.join(DATA_PATH, 'labels')

SENTENCE_EMBEDDING_SUFFIX = '_new_dialogsum_sentence_transformer'
TOPIC_SEGMENTATION_SUFFIX = '_sent_c99_label'
STAGE_SEGMENTATION_SUFFIX = '_sent_trans_cons_label'