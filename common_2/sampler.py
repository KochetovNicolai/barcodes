import os
import pickle
import numpy as np

class Sampler:
    def __init__(self, conf_path, train_ratio=0.8):
        files = np.array(os.listdir(conf_path))
        sample = np.random.sample(len(files))
        train_ind = np.where(sample <= train_ratio)
        test_ind = np.where(sample > train_ratio)
        self.train = files[train_ind]
        self.test = files[test_ind]

    def dump(self):
        pickle.dump(self.train, open("train.p", "wb"))
        pickle.dump(self.test, open("test.p", "wb"))
        print 'dumped', len(self.train), 'train', len(self.test), 'test'

    def load(self):
        self.train = pickle.load(open("train.p", "rb"))
        self.test = pickle.load(open("test.p", "rb"))
        print 'loaded', len(self.train), 'train', len(self.test), 'test'
