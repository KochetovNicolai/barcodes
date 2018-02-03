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

    def dump(self, pref):
        pickle.dump(self.train, open(pref + "_train.p", "wb"))
        pickle.dump(self.test, open(pref + "_test.p", "wb"))
        print 'dumped', len(self.train), 'train', len(self.test), 'test'

    def load(self, pref):
        self.train = pickle.load(open(pref + "_train.p", "rb"))
        self.test = pickle.load(open(pref + "_test.p", "rb"))
        print 'loaded', len(self.train), 'train', len(self.test), 'test'
