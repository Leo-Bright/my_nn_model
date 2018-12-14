#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
from multiprocessing import Process, Pool, Value, Array
import numpy as np
import optparse
import os
import random
import sys
import time
import warnings

from ds import mp


__author__ = 'sheep'


class Common(object):

    def __init__(self):
        self.node_vocab = None
        self.node2vec = None

    def train(self, walks, seed=None):
        raise NotImplementedError

    def dump_to_file(self, output_fname, type_='node'):
        '''
            input:
                type_: 'node' or 'path'
        '''
        with open(output_fname, 'w') as f:
            if type_ == 'node':
                f.write('%d %d\n' % (len(self.node_vocab), self.size))
                for node, vector in zip(self.node_vocab, self.node2vec):
                    line = ' '.join([str(v) for v in vector])
                    f.write('%s %s\n' % (node.node_id, line))
            else:
                f.write('%d %d\n' % (self.path_vocab.distinct_path_count(),
                                     self.size))
                for path, vector in zip(self.path_vocab, self.path2vec):
                    if path.is_inverse:
                        continue
                    line = ' '.join([str(v) for v in vector])
                    f.write('%s %s\n' % (path.path_id, line))


class MP2Vec(Common):

    def __init__(self, size=100, window=10, neg=5,
                       alpha=0.005, num_processes=1, iterations=1,
                       normed=True, same_w=False,
                       is_no_circle_path=True):
        '''
            size:      Dimensionality of word embeddings
            window:    Max window length
            neg:       Number of negative examples (>0) for
                       negative sampling, 0 for hierarchical softmax
            alpha:     Starting learning rate
            num_processes: Number of processes
            iterations: Number of iterations
            normed:    To normalize the final vectors or not
            same_w:    Same matrix for nodes and context nodes
            is_no_circle_path: Generate training data without circle in the path
        '''
        self.size = size
        self.window = window
        self.neg = neg
        self.alpha = alpha
        self.num_processes = num_processes
        self.iterations = iterations
        self.vocab = None
        self.node2vec = None
        self.normed = normed
        self.same_w = same_w
        self.is_no_circle_path = is_no_circle_path

    def train(self, G, training_fname,  seed=None, k_hop_neighbors=None):
        '''
            input:
                walks:
                    each element: [<node_id>, <node_id>, <node_id>,....]
        '''

        def get_training_size(fname):
            with open(fname, 'r') as f:
                for line in f:
                    pass
                return f.tell()

        node_vocab = mp.NodeVocab.load_from_file(training_fname)

        print 'distinct node count: %d' % len(node_vocab)
        training_size = get_training_size(training_fname)
        print 'training walks size: %d' % training_size

        #initialize vectors
        Wx, Wy = MP2Vec.init_net(self.size, len(node_vocab))

        counter = Value('i', 0)
        tables = {
            'all': UnigramTable(node_vocab, uniform=True)
        }

        print 'start training'
        if self.num_processes > 1:
            processes = []
            for i in range(self.num_processes):
                start = training_size / self.num_processes * i
                end = training_size / self.num_processes * (i+1)
                if i == self.num_processes-1:
                    end = training_size

                p = Process(target=train_process,
                                   args=(i, node_vocab,
                                         Wx, Wy, tables,
                                         self.neg, self.alpha,
                                         self.window, counter,
                                         self.iterations,
                                         training_fname, (start, end),
                                         self.same_w,
                                         k_hop_neighbors,
                                         self.is_no_circle_path))
                processes.append(p)

            start = time.time()
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            end = time.time()
        else:
            start = time.time()
            train_process(0, node_vocab,
                          Wx, Wy, tables,
                          self.neg, self.alpha,
                          self.window, counter,
                          self.iterations,
                          training_fname, (0, training_size),
                          self.same_w,
                          k_hop_neighbors,
                          self.is_no_circle_path)
            end = time.time()

        self.node_vocab = node_vocab

        #normalize node and path vectors
        node2vec = []
        if self.normed:
            for vec in Wx:
                vec = np.array(list(vec))
                norm=np.linalg.norm(vec)
                if norm==0:
                    node2vec.append(vec)
                else:
                    node2vec.append(vec/norm)
        else:
            for vec in Wx:
                node2vec.append(np.array(list(vec)))
        self.node2vec = node2vec

        print
        print 'Finished. Total time: %.2f minutes' %  ((end-start)/60)

    @staticmethod
    def load_id2vec(fname):
        id2vec = {}
        with open(fname, 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                tokens = line.strip().split(' ')
                id_ = int(tokens[0])
                id2vec[id_] = map(float, tokens[1:])
        return id2vec

    @staticmethod
    def init_net(dim, node_size, id2vec=None):
        '''
            return
                Wx: a |V|*d matrix for input layer to hidden layer
                Wy: a |V|*d matrix for hidden layer to output layer
                Wpath: a |paths|*d matrix for hidden layer to output layer
        '''
        tmp = np.random.uniform(low=-0.5/dim,
                                high=0.5/dim,
                                size=(node_size, dim)).astype(np.float64)
        Wx = np.ctypeslib.as_ctypes(tmp)
        Wx = Array(Wx._type_, Wx, lock=False)

        if id2vec is not None:
            for i, vec in sorted(id2vec.items()):
                for j in range(len(vec)):
                    Wx[i][j] = vec[j]

        tmp = np.random.uniform(low=-0.5/dim,
                                high=0.5/dim,
                                size=(node_size, dim)).astype(np.float64)
        Wy = np.ctypeslib.as_ctypes(tmp)
        Wy = Array(Wy._type_, Wy, lock=False)

        return Wx, Wy


class UnigramTable(object):
    '''
        For negative sampling.
        A list of indices of words in the vocab
        following a power law distribution.
    '''
    def __init__(self, vocab, seed=None, size=1000000, times=1, node_ids=None, uniform=False):
        self.table = UnigramTable.generate_table(vocab,
                                                 vocab.count() * times,
                                                 node_ids,
                                                 uniform)
        if seed is not None:
            np.random.seed(seed)
        self.randints = np.random.randint(low=0,
                                          high=len(self.table),
                                          size=size)
        self.size = size
        self.index = 0

    @staticmethod
    def generate_table(vocab, table_size, node_ids, uniform):
        power = 0.75
        if node_ids is not None:
            if uniform:
                total = len([t for t in vocab
                             if t.node_id in node_ids])
            else:
                total = sum([math.pow(t.count, power) for t in vocab
                             if t.node_id in node_ids])
        else:
            if uniform:
                total = len(vocab)
            else:
                total = sum([math.pow(t.count, power) for t in vocab])

        table = np.zeros(table_size, dtype=np.uint32)
        p = 0
        current = 0
        for index, word in enumerate(vocab.nodes):
            if node_ids is not None and word.node_id not in node_ids:
                continue

            if uniform:
                p += float(1.0)/total
            else:
                p += float(math.pow(word.count, power))/total

            to_ = int(table_size * p)
            if to_ != table_size:
                to_ = to_+1
            for i in xrange(current, to_):
                table[i] = index
            current = to_
        return table

    def cleanly_sample(self, neighbors, count):
        samples = []
        while True:
            unchecked = self.sample(count)
            for s in unchecked:
                if s in neighbors:
                    continue
                samples.append(s)
                if len(samples) >= count:
                    return samples

    def sample(self, count):
        if count == 0:
            return []

        if self.index + count < self.size:
            samples = [self.table[i] for i
                       in self.randints[self.index:self.index+count]]
            self.index += count
            return samples

        if self.index + count == self.size:
            samples = [self.table[i] for i
                       in self.randints[self.index:]]
            self.index = 0
            self.randints = np.random.randint(low=0,
                                              high=len(self.table),
                                              size=self.size)
            return samples

        self.index = 0
        self.randints = np.random.randint(low=0,
                                          high=len(self.table),
                                          size=self.size)
        return self.sample(count)


#TODO speed up
#TODO the order of edges of the path
def get_context(node_index_walk, index, window_size, no_circle=False):
    start = max(index - window_size, 0)
    end = min(index + window_size + 1, len(node_index_walk))
    context = []
    if no_circle:
        x = node_index_walk[index]
        visited = set()
        for i in range(index+1, end):
            y = node_index_walk[i]
            if x == y or y in visited:
                break
            context.append(y)
            visited.add(y)
    else:
        for i in range(index+1, end):
            context.append(node_index_walk[i])
    return context

def sigmoid(x):
    if x > 6:
        return 1.0
    elif x < -6:
        return 0.0
    return 1 / (1 + math.exp(-x))

def train_process(pid, node_vocab, Wx, Wy,
                  tables,
                  neg, starting_alpha, win, counter,
                  iterations, training_fname, start_end,
                  same_w, k_hop_neighbors,
                  is_no_circle_path):

    def dev_sigmoid(x):
        ex = math.exp(-x)
        s = 1 / (1 + ex)
        return s * (1-s)

    np.seterr(invalid='raise', over ='raise', under='raise')

    #ignore the PEP 3118 buffer warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        Wx = np.ctypeslib.as_array(Wx)
        Wy = np.ctypeslib.as_array(Wy)

    error_fname = 'error.%d' % pid
    os.system('rm -f %s' % error_fname)

    step = 10000
    dim = len(Wx[0])
    alpha = starting_alpha
    start, end = start_end

    table = tables['all']
    cur_win = win

    for iteration in range(iterations):
        word_count = 0

        with open(training_fname, 'r') as f:
            f.seek(start)
            while f.tell() < end:
                #read a random walk
                walk = f.readline().strip().split()
                if len(walk) <= 2:
                    continue

                node_index_walk = [node_vocab.node2index[x]
                                   for i, x in enumerate(walk)]

                for i, x in enumerate(node_index_walk):
                    #generate positive training data
                    for pos_y in get_context(node_index_walk, i, cur_win, no_circle=is_no_circle_path):

                        #generate negative training data
                        if k_hop_neighbors is not None:
                            k_hop_neighbors_index = {}
                            for node_osmid in k_hop_neighbors:
                                k_hop_neighbors_index[node_vocab.node2index[node_osmid]] = []
                                for neighbors_osmid in k_hop_neighbors[node_osmid]:
                                    k_hop_neighbors_index[node_vocab.node2index[node_osmid]].append(node_vocab.node2index[neighbors_osmid])

                            negs = table.cleanly_sample(k_hop_neighbors_index[x], neg)
                        else:
                            negs = table.sample(neg)

                        #SGD learning
                        for y, label in ([(pos_y, 1)] + [(y, 0) for y in negs]):

                            if x == y:
                                continue
                            if label == 0 and y == pos_y:
                                continue

                            wx = Wx[x]
                            if same_w:
                                wy = Wx[y]
                            else:
                                wy = Wy[y]

                            dot = sum(wx * wy)
                            p = sigmoid(dot)
                            g = alpha * (label - p)
                            if g == 0:
                                continue

                            wx += g * wy
                            wy += g * wx


                    word_count += 1

                    if word_count % step == 0:
                        counter.value += step
                        ratio = float(counter.value)/node_vocab.node_count
                        ratio = ratio/iterations

                        alpha = starting_alpha * (1-ratio)
                        if alpha < starting_alpha * 0.0001:
                            alpha = starting_alpha * 0.0001

                        sys.stdout.write(("\r%f "
                                          "%d/%d (%.2f%%) "
                                          "" % (alpha,
                                               counter.value,
                                               node_vocab.node_count*iterations,
                                               ratio*100,
                                               )))
                        sys.stdout.flush()

        counter.value += (word_count % step)
        ratio = float(counter.value)/node_vocab.node_count
