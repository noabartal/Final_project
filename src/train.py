from dkn import DKN
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def get_feed_dict(model, data, start, end):
    feed_dict = {model.clicked_words: data.clicked_words[start:end],
                 model.clicked_entities: data.clicked_entities[start:end],
                 model.news_words: data.news_words[start:end],
                 model.news_entities: data.news_entities[start:end],
                 model.labels: data.labels[start:end],
                 model.users: data.users[start:end]}
    return feed_dict


def train(args, train_data, test_data, kgcn):
    model = DKN(args, kgcn)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for step in range(args.n_epochs):
            # training
            start_list = list(range(0, (train_data.size - train_data.size % args.batch_size), args.batch_size))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + args.batch_size
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, end))
            print("loss: ", loss)
            # evaluation
            # train_auc, train_f1 = model.eval(sess, get_feed_dict(model, train_data, 0, train_data.size))
            test_auc_all, test_f1_all, test_auc_mean_batches, test_f1_mean_batches = eval_batches(sess, model, test_data, args)
            # print('epoch %d    train_auc: %.4f    train_f1: %.4f' % (step, train_auc, train_f1))
            print('epoch %d     test_auc_all: %.4f  test_f1_all: %.4f   test_auc_mean_batches: %.4f    '
                  'test_f1_mean_batches: %.4f' % (step, test_auc_all, test_f1_all, test_auc_mean_batches,
                                                  test_f1_mean_batches))


def eval_batches(sess, model, test_data, args):

    start_list = list(range(0, (test_data.size - test_data.size % args.batch_size), args.batch_size))
    auc_list = []
    f1_list = []
    labels_list = np.array([])
    scores_list = np.array([])
    for start in start_list:
        end = start + args.batch_size
        labels, scores, auc, f1 = model.eval(sess, get_feed_dict(model, test_data, start, end))
        labels_list = np.append(labels_list, labels)
        scores_list = np.append(scores_list, scores)
        if auc is not None:
            auc_list.append(auc)
        f1_list.append(f1)

    auc_all = roc_auc_score(y_true=labels_list, y_score=scores_list)
    scores_list[scores_list > 0.5] = 1
    scores_list[scores_list <= 0.5] = 0
    f1_all = f1_score(y_true=labels_list, y_pred=scores_list)
    return auc_all, f1_all, float(np.mean(auc_list)), float(np.mean(f1_list))

