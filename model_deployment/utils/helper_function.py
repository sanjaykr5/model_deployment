import pickle
import re
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import nltk

nltk.download('stopwords')


class Vocab:
    def __init__(self, path_to_word2idx):
        with open(path_to_word2idx, 'r') as f:
            self.word2idx = json.load(f)

        self.embedding_dim = 100


def collate_fn(batch):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    features, lengths = zip(*batch)
    features = pad_sequence(features, batch_first=True, padding_value=0)
    return features, torch.Tensor(lengths).int()


def tokenize(text):
    tokenized = text.split(' ')
    return tokenized


def return_idx(token, word2idx):
    try:
        return word2idx[token]
    except KeyError:
        return word2idx['<unk>']


def encode_sequence(tokenized, word2idx, max_length=30):
    encoded = np.array([return_idx(token, word2idx) for token in tokenized])
    length = min(max_length, len(encoded))
    return encoded[:length], length


def lower_text(text: str) -> str:
    """
    Convert text to Lowercase text
    :param text: input_text
    :return: output lowercase text
    """
    return text.lower()


def remove_stopwords(text: str) -> str:
    """
    Remove Stopwords from text
    :param text: input_text
    :return: output text without stopwords
    """
    stop = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


def remove_punctuation(text: str) -> str:
    """
    Remove Punctuation and replace it with white space
    :param text: input_text
    :return: output text without punctuation
    """
    return re.sub(r'[^\w\s]', ' ', text)


def preprocess_df(df, column):
    """
    Apply lower text, remove punctuation & remove stopwords on the given column and return dataframe
    :param df:
    :param column:
    :return:
    """
    df[column] = df[column].apply(lambda x: lower_text(x))
    df[column] = df[column].apply(lambda x: remove_punctuation(x))
    df[column] = df[column].apply(lambda x: remove_stopwords(x))
    return df


class TextDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.df.loc[idx, 'encoded_x'][0])
        l = self.df.loc[idx, 'encoded_x'][1]
        return x, l


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_size, num_output):
        super(LSTM, self).__init__()
        self.embeddings_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embeddings_layer.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(2 * hidden_size, num_output)

    def forward(self, text_index, text_lengths):
        embedded = self.embeddings_layer(text_index)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        output = self.dense(self.dropout(hidden))
        return output
