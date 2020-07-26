# split each item in the train set to multiple items according to the enitities that appear in it
def split_by_entities(file, output_file, user_dict):
    reader = open(file, encoding='utf-8')
    writer = open(output_file, 'w', encoding='utf-8')
    # different samples might have the same user-entity combination, we don't need duplicates
    seen_lines = dict()
    # count uniqe entites in train set
    all_entities = dict()
    # map users to indices from 0 to len(users)
    curr_user = 0
    for line in reader:
        user, words_id, entities_id, is_click = line.split('\t')
        if user in user_dict:
            user_idx = user_dict[user]
        else:
            user_idx = curr_user
            curr_user += 1
            user_dict[user] = user_idx
        entities_id = entities_id.split(',')
        unique_entities = list(dict.fromkeys(entities_id))
        for ent in unique_entities:
            all_entities[ent] = 1
            line = '\t'.join([user, ent, is_click])
            if ent != '0' and ent != ',' and line not in seen_lines:
                writer.write(line)
                seen_lines[line] = 1
    reader.close()
    writer.close()
    print("# entities: ", len(all_entities))

# convert entities in the kg to the indices as they appear in train_kgcn and test_kgcn
def kg_to_idx(max_idx, dataset):
    reader = open(dataset + '/kg/kg.txt', encoding='utf-8')
    entity2index = open(dataset + '/kg/entity2index.txt', encoding='utf-8')
    ent_to_idx = dict()
    for line in entity2index:
        ent, idx = line.split('\t')
        idx = idx[:-1] # remove \n at the end
        ent_to_idx[ent] = idx
    entity2index.close()
    writer = open(dataset + '/kg/kg_kgcn.txt', 'w', encoding='utf-8')
    for line in reader:
        entity_1, relation, entity_2 = line.split('\t')
        entity_2 = entity_2[:-1] # remove \n at the end
        if entity_1 in ent_to_idx:
            idx_1 = ent_to_idx[entity_1]
        else:
            idx_1 = str(max_idx)
            max_idx += 1
            ent_to_idx[entity_1] = idx_1
        if entity_2 in ent_to_idx:
            idx_2 = ent_to_idx[entity_2]
        else:
            idx_2 = str(max_idx)
            max_idx += 1
            ent_to_idx[entity_2] = idx_2
        writer.write('%s\t%s\t%s\n' % (idx_1, relation, idx_2))
    writer.close()


user_dict = dict()
split_by_entities('books/train.txt',  'books/train_kgcn.txt', user_dict)
split_by_entities('books/test.txt', 'books/test_kgcn.txt', user_dict)

# kg_to_idx(max_idx = 3777, dataset='news)
