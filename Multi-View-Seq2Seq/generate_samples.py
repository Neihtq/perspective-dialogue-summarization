import os
import torch

from fairseq.models.bart import BARTModel
from rouge import Rouge, FilesRouge
from tqdm import tqdm_notebook as tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

def create_hypotheses(weights_path, data_path, source, source2):
    bart = BARTModel.from_pretrained(
        weights_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=data_path
    )

    bart.eval()
    bart.cuda()
    slines = [source.strip()]
    slines2 = [source2.strip()]
    hypotheses = bart.sample(slines, sentences2 = slines2, balance = True, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=3)
    torch.cuda.empty_cache()
    
    return hypotheses


def generate_n_samples(hyp_path, dest_path, n=20):
    ref_path = './data/dialogsum/DialogSum_Data/test_dialogsum_sent_trans_cons_label_2.target'
    hypothesis = []
    with open(hyp_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                hypothesis.append(l[:-1])

    reference = []
    with open(ref_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            reference.append(l[:-1])

    with open(dest_path, 'w') as file:
        for i in range(n):
            file.write(hypothesis[i] + '\n')

    
def main():
    print('==================Pretrained Model==================')
    pretrained_test_hypo = './data/dialogsum/DialogSum_Data/test_best_multi_attn_best_PRETRAINED.hypo'
    generate_n_samples(pretrained_test_hypo, './dialogsum_pretrained.txt')

    print('==================Single View Model==================')
    test_hypo_single = './data/dialogsum/DialogSum_Data/test_best_multi_attn_best_SINGLE.hypo'
    generate_n_samples(test_hypo_single, './dialogsum_single.txt')

    print('==================Multi View Model==================')
    test_hypo_multi = './data/dialogsum/DialogSum_Data/test_best_multi_attn_best_MULTI.hypo'
    generate_n_samples(test_hypo_multi, './dialogsum_multi.txt')


if __name__ == '__main__':
    main()