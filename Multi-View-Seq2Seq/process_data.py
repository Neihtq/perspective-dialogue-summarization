import os
import torch

from sentence_transformers import SentenceTransformer

from utils.constants import NEW_DIALOGSUM_PATH, NEW_DIALOGSUM_PATH, TOPIC_SEGMENTATION_SUFFIX, STAGE_SEGMENTATION_SUFFIX
from utils.data_preprocess import embed_sentences, segment_topic, segment_stages, transform_format


def embed_data(paths):
    embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    embedder.cuda()
    for path in paths:
        embed_sentences(path, embedder)
    del embedder

    torch.cuda.empty_cache()


def read_labels(paths):
    for path in paths:
        transform_format(path, '_all')
        transform_format(path, '_none')
        transform_format(path, STAGE_SEGMENTATION_SUFFIX)
        transform_format(path, TOPIC_SEGMENTATION_SUFFIX)


if __name__ == '__main__':
    train_path = os.path.join(NEW_DIALOGSUM_PATH, 'train.json')
    val_path = os.path.join(NEW_DIALOGSUM_PATH, 'val.json')
    test_path = os.path.join(NEW_DIALOGSUM_PATH, 'test.json')
    paths = [train_path, val_path, test_path]

    print('Start Sentence Embedding.')
    #embed_data(paths)

    print('Start Topic Segmentation.')
    #for path in paths:
        #segment_topic(path)

    print('Start Stage Segmentation')
    #segment_stages(paths)

    print('Start Rading Labels')
    read_labels(paths)