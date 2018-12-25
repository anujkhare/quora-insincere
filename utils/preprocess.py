from torch.utils.data import Dataset, DataLoader
from typing import Dict
import operator 
import re
import numpy as np
import pandas as pd


def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
#     for punct in '&':
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, f' {punct} ')
#     for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
#         x = x.replace(punct, '')
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium',
                'fortnite': 'fortnight',
                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def preprocessor(text: str) -> str:
    processed = replace_typical_misspell(clean_text(text))
    return processed



class QuoraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, word_vectors: Dict, preprocessor = None, col_label: str = 'target', col_sentence: str = 'question_text') -> None:
        if preprocessor is not None:
            df[col_sentence] = df[col_sentence].progress_apply(preprocessor)

        self.df = df
        self.word_vectors = word_vectors
        
        self.col_label = col_label
        self.col_sentence = col_sentence
        
    def __len__(self) -> int:
        return len(self.df)

    def _sentence_to_vec(self, sentence: str) -> np.ndarray:
        # FIXME
        vectors = []
        final_sentence = []
        for word in sentence.split(' '):
            if word not in self.word_vectors:
#                 vectors.append(np.zeros(300, dtype=np.float32))
                continue

            final_sentence.append(word)
            vectors.append(self.word_vectors[word])

#         vectors = np.vstack(vectors)
        vectors = np.array(vectors)
        final_sentence = ' '.join(final_sentence)
        return vectors, final_sentence
        
    def _preprocess(self, record):
        # Convert sentences to word vectors, return list of 
        v1, fs1 = self._sentence_to_vec(record[self.col_sentence])
        return {
            'sentence1': v1,
            'label': record[self.col_label],
            'final_sentence1': fs1,
        }

    def __getitem__(self, ix):
        return self._preprocess(self.df.iloc[ix])