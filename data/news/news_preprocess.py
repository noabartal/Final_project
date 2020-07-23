import re
import os
import gensim
import numpy as np

PATTERN1 = re.compile('[^A-Za-z]')
PATTERN2 = re.compile('[ ]{2,}')
WORD_FREQ_THRESHOLD = 2
ENTITY_FREQ_THRESHOLD = 1
MAX_TITLE_LENGTH = 10
WORD_EMBEDDING_DIM = 100
DATASET = 'news'

word2freq = {}
entity2freq = {}
word2index = {}
entity2index = {}
corpus = []


def count_word_and_entity_freq(files):
    """
    Count the frequency of words and entities in news titles in the training and test files
    :param files: [training_file, test_file]
    :return: None
    """
    for file in files:
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.strip().split('\t')
            news_title = array[1]
            entities = array[3]

            # count word frequency
            for s in news_title.split(' '):
                if s not in word2freq:
                    word2freq[s] = 1
                else:
                    word2freq[s] += 1

            # count entity frequency
            # if DATASET is not 'books':
            for s in entities.split(';'):
                if DATASET is not 'books':
                    entity_id = s[:s.index(':')]
                else:
                    entity_id = s
                if entity_id not in entity2freq:
                    entity2freq[entity_id] = 1
                else:
                    entity2freq[entity_id] += 1

            corpus.append(news_title.split(' '))
        reader.close()


def construct_word2id_and_entity2id():
    """
    Allocate each valid word and entity a unique index (start from 1)
    :return: None
    """
    cnt = 1  # 0 is for dummy word
    for w, freq in word2freq.items():
        if freq >= WORD_FREQ_THRESHOLD:
            word2index[w] = cnt
            cnt += 1
    print('- word size: %d' % len(word2index))

    if DATASET is not 'books':
        path = 'kg/'
    else:
        path = '../books/kg/'
    writer = open(path + 'entity2index.txt', 'w', encoding='utf-8')
    cnt = 1
    for entity, freq in entity2freq.items():
        if freq >= ENTITY_FREQ_THRESHOLD:
            entity2index[entity] = cnt
            writer.write('%s\t%d\n' % (entity, cnt))  # for later use
            cnt += 1
    writer.close()
    print('- entity size: %d' % len(entity2index))


def get_local_word2entity(entities):
    """
    Given the entities information in one line of the dataset, construct a map from word to entity index
    E.g., given entities = 'id_1:Harry Potter;id_2:England', return a map = {'harry':index_of(id_1),
    'potter':index_of(id_1), 'england': index_of(id_2)}
    :param entities: entities information in one line of the dataset
    :return: a local map from word to entity index
    """
    local_map = {}

    for entity_pair in entities.split(';'):
        entity_id = entity_pair[:entity_pair.index(':')]
        entity_name = entity_pair[entity_pair.index(':') + 1:]

        # remove non-character word and transform words to lower case
        entity_name = PATTERN1.sub(' ', entity_name)
        entity_name = PATTERN2.sub(' ', entity_name).lower()

        # constructing map: word -> entity_index
        for w in entity_name.split(' '):
            entity_index = entity2index[entity_id]
            local_map[w] = entity_index

    return local_map


def encoding_title(title, entities):
    """
    Encoding a title according to word2index map and entity2index map
    :param title: a piece of news title
    :param entities: entities contained in the news title
    :return: encodings of the title with respect to word and entity, respectively
    """
    if DATASET is not 'books':
        local_map = get_local_word2entity(entities)

    # local_map = get_local_word2entity(entities)
    array = title.split(' ')
    word_encoding = ['0'] * MAX_TITLE_LENGTH
    entity_encoding = ['0'] * MAX_TITLE_LENGTH

    point = 0
    for s in array:
        if s in word2index:
            word_encoding[point] = str(word2index[s])
            if DATASET is not 'books' and s in local_map:
                if s in local_map:
                    entity_encoding[point] = str(local_map[s])
            point += 1
        if point == MAX_TITLE_LENGTH:
            break
    word_encoding = ','.join(word_encoding)
    if DATASET is 'books':
        entity_encoding[0] = str(entities)
    entity_encoding = ','.join(entity_encoding)
    return word_encoding, entity_encoding


def transform(input_file, output_file, id_to_index):
    reader = open(input_file, encoding='utf-8')
    writer = open(output_file, 'w', encoding='utf-8')
    for line in reader:
        array = line.strip().split('\t')
        user_id = id_to_index[array[0]]
        title = array[1]
        label = array[2]
        entities = array[3]
        word_encoding, entity_encoding = encoding_title(title, entities)
        writer.write('%s\t%s\t%s\t%s\n' % (user_id, word_encoding, entity_encoding, label))
    reader.close()
    writer.close()


def get_word2vec_model():
    if not os.path.exists('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model'):
        print('- training word2vec model...')
        w2v_model = gensim.models.Word2Vec(corpus, size=WORD_EMBEDDING_DIM, min_count=1, workers=16)
        print('- saving model ...')
        w2v_model.save('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    else:
        print('- loading model ...')
        w2v_model = gensim.models.word2vec.Word2Vec.load('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    return w2v_model


def user_id_to_index(train, test):
    reader_train = open(train, encoding='utf-8')
    id_to_index = dict()
    index = 1
    for line in reader_train:
        array = line.strip().split('\t')
        user_id = array[0]
        if user_id not in id_to_index:
            id_to_index[user_id] = index
            index += 1
    reader_train.close()
    reader_test = open(test, encoding='utf-8')
    for line in reader_test:
        array = line.strip().split('\t')
        user_id = array[0]
        if user_id not in id_to_index:
            id_to_index[user_id] = index
            index += 1
    reader_test.close()
    return id_to_index


if __name__ == '__main__':
    path = '../' + DATASET
    print('counting frequencies of words and entities ...')
    count_word_and_entity_freq([path+'/raw_train.txt', path+'/raw_test.txt'])

    print('constructing word2id map and entity to id map ...')
    construct_word2id_and_entity2id()

    id_to_index = user_id_to_index(path+'/raw_train.txt', path+'/raw_test.txt')

    print('transforming training and test dataset ...')
    transform(path+'/raw_train.txt', path+'/train.txt', id_to_index)
    transform(path+'/raw_test.txt', path+'/test.txt', id_to_index)

    print('getting word embeddings ...')
    embeddings = np.zeros([len(word2index) + 1, WORD_EMBEDDING_DIM])
    model = get_word2vec_model()
    for index, word in enumerate(word2index.keys()):
        embedding = model[word] if word in model.wv.vocab else np.zeros(WORD_EMBEDDING_DIM)
        embeddings[index + 1] = embedding
    print('- writing word embeddings ...')
    np.save((path+'/word_embeddings_' + str(WORD_EMBEDDING_DIM)), embeddings)
