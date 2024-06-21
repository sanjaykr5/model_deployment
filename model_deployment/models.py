import json
import os
import pickle
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
import textdistance as td

from .utils.helper_function import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ml_model_pipeline(text):
    text = lower_text(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text


def lstm_model_pipeline(text, vocab):
    text = tokenize(text)
    text = encode_sequence(text, vocab.word2idx)
    x, l = torch.from_numpy(text[0]), text[1]
    l = [l]
    x = x.unsqueeze(0)
    return x, l


def load_processing_variables():
    with open(f'model_files/count_vectorizer.pkl', 'rb') as f:
        count_vec = pickle.load(f)
    with open(f'model_files/class_names.pkl', 'rb') as f:
        lb = pickle.load(f)
    vocab = Vocab('model_files/word2idx.json')
    return count_vec, lb, vocab


class ClassificationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.count_vec, self.lb, self.vocab = load_processing_variables()
        self.load_model(self.model_name)

    def load_model(self, model_name):
        if model_name in ['lr', 'svm', 'nb', 'rf', 'xgb']:
            with(open(f'model_files/{model_name}.pkl', 'rb')) as f:
                self.model = pickle.load(f)
        else:
            print('error', model_name)
            self.model = LSTM(len(self.vocab.word2idx.keys()), self.vocab.embedding_dim, None, 100, num_output=10).to(
                device)
            state_dict = torch.load('model_files/lstm.tar')
            self.model.load_state_dict(state_dict)
        print(f'successfully loaded {model_name}')

    def change_model(self, model_name):
        self.model_name = model_name
        self.load_model(model_name)

    def predict(self, text):
        if self.model_name in ['lr', 'svm', 'nb', 'rf', 'xgb']:
            text = ml_model_pipeline(text)
            text = np.array(self.count_vec.transform([text]).todense())
            pred = self.model.predict(text)
            pred = self.lb.inverse_transform(pred)
            return pred
        else:
            x, l = lstm_model_pipeline(text, self.vocab)
            _, pred = torch.max(self.model(x.to(device), l), 1)
            pred = self.lb.inverse_transform(pred.detach().cpu().tolist())
            return pred


def load_cluster_mapping(path_to_json):
    with open(path_to_json, 'r') as f:
        cluster_label_mapping = json.load(f)
    return cluster_label_mapping


def search_token_tokenized(word, cluster_label):
    if word in cluster_label.split(' '):
        return True
    return False


def search_similarity_tokenized(text, cluster_labels):
    match_scores = np.zeros(len(cluster_labels))
    search_fun = np.vectorize(search_token_tokenized)
    for word in text.split(' '):
        match_scores += search_fun(word, cluster_labels)
    return cluster_labels[np.argmax(match_scores)]


def cosine_similarity(text, cluster_label):
    return td.cosine(text.split(' '), cluster_label.split(' '))


def search_max_cosine(text, cluster_labels):
    cos_fun = np.vectorize(cosine_similarity)
    match_scores = cos_fun(text, cluster_labels)
    return cluster_labels[np.argmax(match_scores)]


class ClusteringModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.cluster_label_mapping = load_cluster_mapping('model_files/cluster_label_mapping.json')
        self.cluster_label_list = list(self.cluster_label_mapping.keys())

    def change_model(self, model_name):
        self.model_name = model_name

    def predict(self, text):
        text = ml_model_pipeline(text)
        if self.model_name == 'sm':
            cluster_label = search_similarity_tokenized(text, self.cluster_label_list)
        else:
            cluster_label = search_max_cosine(text, self.cluster_label_list)
        return self.cluster_label_mapping[cluster_label]
