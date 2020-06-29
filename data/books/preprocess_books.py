import argparse
import numpy as np
import re

RATING_FILE_NAME = dict({'movie': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie': ',', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0})

def read_item_index_to_entity_id_file():
    file = 'item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1

def map_id_to_title():
    file = 'BX-Books.csv'
    for line in open(file).readlines()[1:]:
        array = line.strip().split(SEP[DATASET])
        array = list(map(lambda x: x[1:-1], array))
        item_index = array[0]
        title = array[1]
        title = re.sub(r'[^\w\s]', '', title)
        title = title.lower()
        if item_index in item_index_old2new:
            # if item_index == '0671021346':
            #     print(item_index_old2new[item_index])
            id_to_title[str(item_index_old2new[item_index])] = title

def add_title_column():
    file = 'ratings_final.txt'
    writer_train = open('raw_train.txt', 'w', encoding='utf-8')
    writer_test = open('raw_test.txt', 'w', encoding='utf-8')
    reader = open(file, 'r', encoding='utf-8')
    ratings_len = 0
    for line in reader.readlines():
        ratings_len +=1
    idx = np.random.choice(ratings_len, int(ratings_len*0.15), replace=True).tolist()
    test_idx = dict()
    for i in idx:
        test_idx[i] = 1
    reader = open(file, 'r', encoding='utf-8')
    for i, line in enumerate(reader.readlines()):
        user_index = line.strip().split('\t')[0]
        item_id = line.strip().split('\t')[1]
        rating = line.strip().split('\t')[2]
        title = id_to_title[item_id]
        if i in test_idx:
            writer_test.write('%s\t%s\t%s\t%s\n' % (user_index, title, rating, item_id))
        else:
            writer_train.write('%s\t%s\t%s\t%s\n' % (user_index, title, rating, item_id))
    writer_train.close()
    writer_test.close()

if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='book', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d

    entity_id2index = dict()
    # relation_id2index = dict()
    item_index_old2new = dict()
    id_to_title = dict()

    read_item_index_to_entity_id_file()
    map_id_to_title()
    add_title_column()

    print('done')


