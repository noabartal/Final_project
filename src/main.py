import argparse
from data_loader import load_data
from train import train
import sys
# sys.path.insert(0, '../KGCN/')
sys.path.append('/../')
# import [file]
# from KGCN import model
# sys.path.insert(1, '../')
# from model import KGCN
import KGCN.data_loader as kgcn_data_loader
from KGCN.model import KGCN
import os
# DATASET = 'books'
DATASET = 'news'
parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='../data/' + DATASET +'/train.txt', help='path to the training file')
parser.add_argument('--test_file', type=str, default='../data/' + DATASET +'/test.txt', help='path to the test file')
parser.add_argument('--transform', type=bool, default=True, help='whether to transform entity embeddings')
# parser.add_argument('--use_context', type=bool, default=True, help='whether to use context embeddings')
parser.add_argument('--dataset', type=str, default=DATASET, help='dataset folder name')
parser.add_argument('--use_context', type=bool, default=False, help='whether to use context embeddings')
parser.add_argument('--max_click_history', type=int, default=30, help='number of sampled click history for each user')
parser.add_argument('--n_filters', type=int, default=32, help='number of filters for each size in KCNN')
parser.add_argument('--filter_sizes', type=int, default=[2, 3], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 2 3')
parser.add_argument('--l2_weight', type=float, default=0.00001, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='number of samples in one batch')
parser.add_argument('--n_epochs', type=int, default=15, help='number of training epochs')
parser.add_argument('--KGE', type=str, default='kgcn',
                    help='knowledge graph embedding method, please ensure that the specified input file exists')
parser.add_argument('--entity_dim', type=int, default=64,
                    help='dimension of entity embeddings, please ensure that the specified input file exists')
parser.add_argument('--word_dim', type=int, default=50,
                    help='dimension of word embeddings, please ensure that the specified input file exists')
parser.add_argument('--max_title_length', type=int, default=10,
                    help='maximum length of news titles, should be in accordance with the input datasets')
args = parser.parse_args()


# parser.add_argument('--train_file', type=str, default='../data/news/train.txt', help='path to the training file')
# parser.add_argument('--test_file', type=str, default='../data/news/test.txt', help='path to the test file')
# parser.add_argument('--transform', type=bool, default=True, help='whether to transform entity embeddings')
# parser.add_argument('--use_context', type=bool, default=True, help='whether to use context embeddings')
# parser.add_argument('--max_click_history', type=int, default=30, help='number of sampled click history for each user')
# parser.add_argument('--n_filters', type=int, default=32, help='number of filters for each size in KCNN')
# parser.add_argument('--filter_sizes', type=int, default=[2, 3], nargs='+',
#                     help='list of filter sizes, e.g., --filter_sizes 2 3')
# parser.add_argument('--l2_weight', type=float, default=0.01, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--batch_size', type=int, default=128, help='number of samples in one batch')
# parser.add_argument('--n_epochs', type=int, default=15, help='number of training epochs')
# parser.add_argument('--KGE', type=str, default='kgcn',
#                     help='knowledge graph embedding method, please ensure that the specified input file exists')
# parser.add_argument('--entity_dim', type=int, default=64,
#                     help='dimension of entity embeddings, please ensure that the specified input file exists')
# parser.add_argument('--word_dim', type=int, default=100,
#                     help='dimension of word embeddings, please ensure that the specified input file exists')
# parser.add_argument('--max_title_length', type=int, default=10,
#                     help='maximum length of news titles, should be in accordance with the input datasets')
# args = parser.parse_args()

# trained KGCN arguments (used only for calculating user-specific entity embeddings)
kgcn_parser = argparse.ArgumentParser()
kgcn_parser.add_argument('--dataset', type=str, default='books', help='which dataset to use')
kgcn_parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
kgcn_parser.add_argument('--n_epochs', type=int, default=1, help='the number of epochs')
kgcn_parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
kgcn_parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
kgcn_parser.add_argument('--n_iter', type=int, default=3, help='number of iterations when computing entity representation')
kgcn_parser.add_argument('--batch_size', type=int, default=128, help='batch size')
kgcn_parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
kgcn_parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
kgcn_parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
kgcn_args = kgcn_parser.parse_args()

# KGCN
data = kgcn_data_loader.load_data(kgcn_args)
n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
train_data, eval_data, test_data = data[4], data[5], data[6]
adj_entity, adj_relation = data[7], data[8]

kgcn = KGCN(kgcn_args, n_user, n_entity, n_relation, adj_entity, adj_relation)
kgcn.load_pretrained_weights()

# DKN
train_data, test_data = load_data(args)
train(args, train_data, test_data, kgcn)
