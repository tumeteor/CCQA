import pickle
from tqdm import tqdm
import random
import numpy as np
from model import ccrc_model


class Config(object):
    emb_dim = 300  # 300d glove embedding
    hidden_dim = 150
    degree = 2
    num_epochs = 20
    early_stopping = 2
    dropout = 0.5
    lr = 0.05
    num_emb = 1000  # default
    emb_lr = 0.1
    reg = 0.0001
    batch_size = 20
    maxseqlen = 100
    maxnodesize = 300
    trainable_embeddings = True
    max_candidate_answers = 100
    word2idx = None


def load_embedding():
    word2idx = {}
    embeddings = []
    with open('/home/tunguyen/CQA/glove/glove.6B.300d.txt', encoding='utf-8') as infile:
        for line in tqdm(infile.readlines()):
            array = line.lstrip().rstrip().split(' ')
            vector = list(map(float, array[1:]))
            embeddings.append(vector)
            word = array[0].lower()
            word2idx[word] = len(embeddings) - 1
    return word2idx, embeddings


def get_preprocess_data():
    with open('preprocess.data', 'rb') as file:
        import time
        t = time.time()
        squad_data = pickle.load(file)
        t1 = time.time()
        word2idx, embeddings = load_embedding()
        t2 = time.time()

        print("load data time:{}, embedding time:{}".format(t1 - t, t2 - t1))
        return squad_data, word2idx, embeddings


def train(restore=False):
    data, word2idx, embedding = get_preprocess_data()
    config = Config()
    config.embedding = embedding
    config.word2idx = word2idx
    assert len(config.word2idx) == len(config.embedding)
    random.seed(42)
    np.random.seed(42)
    config.num_emb = len(word2idx)

    model = ccrc_model(config)

    model.train(data['train'], restore)


if __name__ == '__main__':
    train(False)
