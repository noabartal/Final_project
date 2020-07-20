import tensorflow as tf
from abc import abstractmethod
import numpy as np

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            print("mix neighb: ", self.batch_size, self.dim)

            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
            # user_embeddings = tf.Print(user_embeddings, [user_embeddings], 'user_embedding_mix meighbor vector')
            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None,
                 load_pretrained=False, iter=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            if load_pretrained:
                weights = np.load(
                    '../KGCN/kgcn_agg_weights_'+str(iter)+'_64_books_2' + '.npy')
                bias = np.load(
                    '../KGCN/kgcn_gg_bias_'+str(iter)+'_64_books_2' + '.npy')
                self.weights = tf.Variable(weights, dtype=np.float32, name='weights')
                self.bias = tf.Variable(bias, dtype=np.float32, name='bias')
            else:
                self.weights = tf.get_variable(
                    shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
                self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        print("=================")
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        # neighbors_agg = tf.Print(neighbors_agg.shape, [neighbors_agg.shape], "output reshaped: ", neighbors_agg.shape)
        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        # output = tf.Print(output.shape, [output.shape], "output reshaped: ", output.shape)
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)
