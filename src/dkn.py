import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
CALC_EMB_PER_USER = False
# from main import DATASET
# DATASET = 'books'

class DKN(object):
    def __init__(self, args, kgcn):
        self.params = []  # for computing regularization loss
        self._build_inputs(args)
        self._build_model(args, kgcn)
        self._build_train(args)

    def _build_inputs(self, args):
        with tf.name_scope('input'):
            self.clicked_words = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, args.max_title_length], name='clicked_words')
            self.clicked_entities = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, args.max_title_length], name='clicked_entities')
            self.news_words = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_title_length], name='news_words')
            self.news_entities = tf.placeholder(
                dtype=tf.int32, shape=[None, args.max_title_length], name='news_entities')
            self.labels = tf.placeholder(
                dtype=tf.float32, shape=[None], name='labels')
            self.users = tf.placeholder(
                dtype=tf.int32, shape=[None], name='users')

    def _build_model(self, args, kgcn):
        with tf.name_scope('embedding'):
            word_embs = np.load('../data/' + args.dataset + '/word_embeddings_' + str(args.word_dim) + '.npy')
            if not CALC_EMB_PER_USER:
               entity_embs = np.load('../data/' + args.dataset + '/kg/entity_embeddings_' + args.KGE + '_' + str(args.entity_dim) + '.npy')
               self.entity_embeddings = tf.Variable(entity_embs, dtype=np.float32, name='entity')
               self.params.append(self.entity_embeddings)

            self.word_embeddings = tf.Variable(word_embs, dtype=np.float32, name='word')
            self.params.append(self.word_embeddings)

            self.kgcn = kgcn

            if args.use_context:
                context_embs = np.load(
                    '../data/'+args.dataset+'/kg/context_embeddings_' + args.KGE + '_' + str(args.entity_dim) + '.npy')
                self.context_embeddings = tf.Variable(context_embs, dtype=np.float32, name='context')
                self.params.append(self.context_embeddings)

            # if args.transform:
            #     self.entity_embeddings = tf.layers.dense(
            #         self.entity_embeddings, units=args.entity_dim, activation=tf.nn.tanh, name='transformed_entity',
            #         kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2_weight))
            #     if args.use_context:
            #         self.context_embeddings = tf.layers.dense(
            #             self.context_embeddings, units=args.entity_dim, activation=tf.nn.tanh,
            #             name='transformed_context', kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2_weight))

        user_embeddings, news_embeddings = self._attention(args)
        self.scores_unnormalized = tf.reduce_sum(user_embeddings * news_embeddings, axis=1)
        self.scores = tf.sigmoid(self.scores_unnormalized)

    def _attention(self, args):
        # (batch_size * max_click_history, max_title_length)
        print("self.clicked_words:", self.clicked_words)
        clicked_words = tf.reshape(self.clicked_words, shape=[-1, args.max_title_length])
        clicked_entities = tf.reshape(self.clicked_entities, shape=[-1, args.max_title_length])
        print("reshaped clicked_words:", clicked_words)
        with tf.variable_scope('kcnn', reuse=tf.AUTO_REUSE):  # reuse the variables of KCNN
            # (batch_size * max_click_history, title_embedding_length)
            # title_embedding_length = n_filters_for_each_size * n_filter_sizes
            clicked_embeddings = self._kcnn(clicked_words, clicked_entities, args, clicked=True)
            print("clicked_embeddings: ", clicked_embeddings.shape)

            # (batch_size, title_embedding_length)
            news_embeddings = self._kcnn(self.news_words, self.news_entities, args)
        print("news_embeddings: ", news_embeddings.shape)

        # (batch_size, max_click_history, title_embedding_length)
        clicked_embeddings = tf.reshape(
            clicked_embeddings, shape=[-1, args.max_click_history, args.n_filters * len(args.filter_sizes)])
        print("clicked_embeddings after reshape: ", clicked_embeddings.shape)

        # (batch_size, 1, title_embedding_length)
        news_embeddings_expanded = tf.expand_dims(news_embeddings, 1)

        # (batch_size, max_click_history)
        attention_weights = tf.reduce_sum(clicked_embeddings * news_embeddings_expanded, axis=-1)

        # (batch_size, max_click_history)
        attention_weights = tf.nn.softmax(attention_weights, dim=-1)

        # (batch_size, max_click_history, 1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

        # (batch_size, title_embedding_length)
        user_embeddings = tf.reduce_sum(clicked_embeddings * attention_weights_expanded, axis=1)

        return user_embeddings, news_embeddings

    def _kcnn(self, words, entities, args, clicked=False):
        # (batch_size * max_click_history, max_title_length, word_dim) for users
        # (batch_size, max_title_length, word_dim) for news
        embedded_words = tf.nn.embedding_lookup(self.word_embeddings, words)
        print("entities: ", entities.shape)
        # print("self.entity_embeddings: ", self.entity_embeddings.shape)
        if not CALC_EMB_PER_USER:
            embedded_entities = tf.nn.embedding_lookup(self.entity_embeddings, entities)
            print("embedded_entities: ", embedded_entities.shape)
            # embedded_entities = tf.Print(embedded_entities, [embedded_entities[1]], message = "embedded_entities val")
        else:
            entities = entities[:, 0]
            print("entities: ", entities.shape)
            print("self.users: ", self.users.shape)
            if clicked: # reshape the users to (batch_size*history_length, title_length)
                repeats = np.full(args.batch_size, args.max_click_history)
                print(repeats)
                users = tf.repeat(self.users, repeats=repeats, axis=0)
                print("users: ", users.shape)
                batches_users = tf.split(users, args.max_click_history, axis=0)
                batches_entities = tf.split(entities, args.max_click_history, axis=0)
                print("batches_users: ", batches_users[0].shape)
                print("batches_entities: ", batches_entities[0].shape)
                print("len batches_users: ", len(batches_users))
                print("len batches_entities: ", len(batches_entities))
                embeddings_list = []
                for users, _entities in zip(batches_users, batches_entities):
                    embeddings_list.append(self.kgcn.get_entity_user_vector(users, _entities))
                    embedded_entities = tf.concat(embeddings_list, axis=0)
            else:
                embedded_entities = self.kgcn.get_entity_user_vector(self.users, entities)
                # TODO: what happens if there are multiple entities per sample?
                # TODO: etract non-zero values from the entities vector, send them separately
                # TODO: to KGCN, and concat the result back to a 3D tensor
            print("embedded_entities: ", embedded_entities.shape)
            # embedded_entities = tf.reshape(embedded_entities, [None, args.entity_dim])
            # print("embedded_entities: ", embedded_entities.shape)
            embedded_entities = tf.expand_dims(embedded_entities, 1)
            paddings = [[0, 0], [0, args.max_title_length-1], [0, 0]]
            embedded_entities = tf.pad(embedded_entities, paddings, 'CONSTANT', constant_values=0)
            # embedded_entities = tf.reshape(embedded_entities, [-1, args.max_title_length, args.entity_dim])
        if args.transform:
            embedded_entities = tf.layers.dense(
                embedded_entities, units=args.entity_dim, activation=tf.nn.tanh, name='transformed_entity',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2_weight))
        print("embedded_entities shape: ", embedded_entities.shape)
        # (batch_size * max_click_history, max_title_length, full_dim) for users
        # (batch_size, max_title_length, full_dim) for news
        if args.use_context:
            embedded_contexts = tf.nn.embedding_lookup(self.context_embeddings, entities)
            concat_input = tf.concat([embedded_words, embedded_entities, embedded_contexts], axis=-1)
            full_dim = args.word_dim + args.entity_dim * 2
        else:
            concat_input = tf.concat([embedded_words, embedded_entities], axis=-1)
            print("concat_input: ", concat_input.shape)
            full_dim = args.word_dim + args.entity_dim

        # (batch_size * max_click_history, max_title_length, full_dim, 1) for users
        # (batch_size, max_title_length, full_dim, 1) for news
        concat_input = tf.expand_dims(concat_input, -1)
        print("concat_input extended: ", concat_input.shape)

        outputs = []
        for filter_size in args.filter_sizes:
            filter_shape = [filter_size, full_dim, 1, args.n_filters]
            w = tf.get_variable(name='w_' + str(filter_size), shape=filter_shape, dtype=tf.float32)
            b = tf.get_variable(name='b_' + str(filter_size), shape=[args.n_filters], dtype=tf.float32)

            if w not in self.params:
                self.params.append(w)

            # (batch_size * max_click_history, max_title_length - filter_size + 1, 1, n_filters_for_each_size) for users
            # (batch_size, max_title_length - filter_size + 1, 1, n_filters_for_each_size) for news
            conv = tf.nn.conv2d(concat_input, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            print("relu: ", relu.shape)

            # (batch_size * max_click_history, 1, 1, n_filters_for_each_size) for users
            # (batch_size, 1, 1, n_filters_for_each_size) for news
            pool = tf.nn.max_pool(relu, ksize=[1, args.max_title_length - filter_size + 1, 1, 1],
                                  strides=[1, 1, 1, 1], padding='VALID', name='pool')
            print("pool: ", pool.shape)
            outputs.append(pool)

        # (batch_size * max_click_history, 1, 1, n_filters_for_each_size * n_filter_sizes) for users
        # (batch_size, 1, 1, n_filters_for_each_size * n_filter_sizes) for news
        output = tf.concat(outputs, axis=-1)
        print("output: ", output.shape)

        # (batch_size * max_click_history, n_filters_for_each_size * n_filter_sizes) for users
        # (batch_size, n_filters_for_each_size * n_filter_sizes) for news
        output = tf.reshape(output, [-1, args.n_filters * len(args.filter_sizes)])
        print("output reshaped: ", output.shape)
        return output

    def _build_train(self, args):
        with tf.name_scope('train'):
            self.base_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores_unnormalized))
            self.l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
            for param in self.params:
                self.l2_loss = tf.add(self.l2_loss, args.l2_weight * tf.nn.l2_loss(param))
            if args.transform:
                self.l2_loss = tf.add(self.l2_loss, tf.losses.get_regularization_loss())
            self.loss = self.base_loss + self.l2_loss
            self.optimizer = tf.train.AdamOptimizer(args.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores > 0.5] = 1
        scores[scores <= 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1
