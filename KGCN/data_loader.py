import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)

    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    train_file = '../data/' + args.dataset + '/train_kgcn'
    test_file = '../data/' + args.dataset + '/test_kgcn'
    if os.path.exists(train_file + '.npy') and os.path.exists(test_file + '.npy'):
        train_data = np.load(train_file + '.npy')
        test_data = np.load(test_file + '.npy')
    else:
        train_data = np.loadtxt(train_file + '.txt', dtype=np.int64)
        test_data = np.loadtxt(test_file + '.txt', dtype=np.int64)
        np.save(train_file + '.npy', train_data)
        np.save(test_file + '.npy', test_data)
        # else:
        #     split_test = True
    train_data, eval_data, test = dataset_split(train_data, args, split_test=False)
    users_train = set(train_data[:, 0])
    items_train = set(train_data[:, 1])
    users_eval = set(eval_data[:, 0])
    items_eval = set(eval_data[:, 1])
    users_test = set(test_data[:, 0])
    items_test = set(test_data[:, 1])
    n_user = len((users_train|users_eval)|users_test)
    n_item = len((items_train|items_eval)|items_test)
    print("n_users: ", n_user)
    print("n_items: ", n_item)
    return n_user, n_item, train_data, eval_data, test_data

# split train to val + train
def dataset_split(rating_np, args, split_test=True):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    train_indices = list(set(range(n_ratings)) - set(eval_indices))
    if split_test:
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    if split_test:
        test_data = rating_np[test_indices]
    else:
        test_data = []

    return train_data, eval_data , test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg/kg_kgcn'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    all_relations = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
        all_relations[relation] = 1
    print("relations num: ", len(all_relations))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    num_ent = 0
    missing_ent = 0
    for entity in range(1, entity_num):
        if entity in kg:
            neighbors = kg[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= args.neighbor_sample_size:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
        else:
            print(entity, " is missing!")
            missing_ent += 1
        num_ent += 1
    print("num ents: ", num_ent, " missing ent: ", missing_ent)
    return adj_entity, adj_relation


