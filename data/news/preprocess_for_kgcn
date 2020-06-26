MAX_IDX = 3777

# split each item in the train set to multiple items according to the enitities that appear in it
def split_by_entities(file, output_file):
    reader = open(file, encoding='utf-8')
    writer = open(output_file, 'w', encoding='utf-8')
    # different samples might have the same user-entity combination, we don't need duplicates
    seen_lines = dict()
    # count uniqe entites in train set
    all_entities = dict()
    for line in reader:
        user, words_id, entities_id, is_click = line.split('\t')
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
    print("# entities in train: ", len(all_entities))

# convert entities in the kg to the indices as they appear in train_kgcn and test_kgcn
def kg_to_idx():
    reader = open('../kg/kg.txt', encoding='utf-8')
    entity2index = open('../kg/entity2index.txt', encoding='utf-8')
    ent_to_idx = dict()
    for line in entity2index:
        ent, idx = line.split('\t')
        idx = idx[:-1] # remove \n at the end
        ent_to_idx[ent] = idx
    entity2index.close()
    writer = open('../kg/kg_kgcn.txt', 'w', encoding='utf-8')
    max_idx = MAX_IDX
    for line in reader:
        entity_1, relation, entity_2 = line.split('\t')
        entity_2 = entity_2[:-1] # remove \n at the end
        # print(entity_1, " ", relation, " ", entity_2)
        if entity_1 in ent_to_idx:
            idx_1 = ent_to_idx[entity_1]
        else:
            idx_1 = str(max_idx)
            # print("entity_1: ", entity_1, " idx_1 :", idx_1)
            max_idx += 1
            ent_to_idx[entity_1] = idx_1
        if entity_2 in ent_to_idx:
            idx_2 = ent_to_idx[entity_2]
        else:
            idx_2 = str(max_idx)
            # print("entity_2: ", entity_2, " idx_2 :", idx_2)
            max_idx += 1
            ent_to_idx[entity_2] = idx_2
        writer.write('%s\t%s\t%s\n' % (idx_1, relation, idx_2))
    writer.close()

# split_by_entities('train.txt', 'train_kgcn.txt')
# split_by_entities('test.txt', 'test_kgcn.txt')

# kg_to_idx()
