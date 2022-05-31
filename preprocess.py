#!/usr/bin/env python3
import sys
import tqdm
import json
from shutil import rmtree
from collections import defaultdict

task = sys.argv[1]
if task == 'wos':
    files=  ['./data/WebOfScience/wos_train.json',
             './data/WebOfScience/wos_val.json',
             './data/WebOfScience/wos_test.json']

    datasets = {}
    label_sets = set()
    for i in files:
        datasets[i] = [json.loads(f) for f in open(i)]
        label_sets |= set(sum([j['doc_label'] for j in datasets[i]], []))

    label_list = sorted(list(label_sets))
    label_map = {label: f'[A_{i}]' for i, label in enumerate(label_list)}

    import pickle
    with open('./data/WebOfScience/label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    def label_to_tgt(labels):
        labels = [label_map[i] for i in labels]
        return ' '.join(labels)

    def label_to_tgt_list(labels):
        labels = [label_map[i] for i in labels]
        return [[i] for i in labels]


    for file_name in datasets:
        assert '.json' in file_name
        if 'train' in file_name:
            with open(file_name.replace('.json', '_generated_tl.json'), 'w') as f:
                for l in datasets[file_name]:
                    f.write(json.dumps({'src': l['doc_token'],
                                        'tgt': label_to_tgt_list(l['doc_label']) }) + '\n')
        else:
            with open(file_name.replace('.json', '_generated.json'), 'w') as f:
                for l in datasets[file_name]:
                    f.write(json.dumps({'src': l['doc_token'],
                                        'tgt': label_to_tgt(l['doc_label']) }) + '\n')

elif task == 'rcv1':
    files = ['./data/rcv1/rcv1_val_all.json',
     './data/rcv1/rcv1_train_all.json',
     './data/rcv1/rcv1_test_all.json']


    label_dict = {}
    hiera = defaultdict(set)
    with open('./data/rcv1/rcv1.taxonomy') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')



    class_label_dict = {}
    def loop_hiera(i, n):
        for j in list(hiera[i]):
            class_label_dict[j] = n
            if j in hiera:
                loop_hiera(j, n + 1)
    loop_hiera(-1, 0)
    print('end')

    d = {}
    label_sets = {}
    for i in files:
        print(i)
        d[i] = [json.loads(f) for f in tqdm.tqdm(open(i))]

        for j in d[i]:
            for l in j['label']:
                label_sets[l] = 0
    label_sets = label_sets.keys()

    label_list = sorted(list(label_sets))

    assert len(label_list) == len(label_dict) == len(class_label_dict)

    label_map = {label: f'[A_{i}]' for i, label in enumerate(label_list)}

    import pickle
    with open('./data/rcv1/label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)


    label_lens = []
    def label_to_tgt(labels):
        global label_lens
        index = [class_label_dict[label_dict[i]] for i in labels]

        labels = [label_map[i] for i in labels]

        ms = sorted(list(zip(index, labels)), key=lambda x: x[0])
        labels = [i[1] for i in ms]

        label_lens.append(len(labels))
        return ' '.join(labels)

    def label_to_tgt_list(labels):
        nyts = [[] for i in range(5)]

        index = [class_label_dict[label_dict[i]] for i in labels]
        labels = [label_map[i] for i in labels]
        for i, l in zip(index, labels):
            nyts[i].append(l)
        # print(''.join([str(len(i)) for i in nyts]))
        return nyts


    for file in d:
        assert '.json' in file
        print(file)
        if 'train' in file:
            with open(file.replace('.json', '_generated_tl.json'), 'w') as f:
                for l in tqdm.tqdm(d[file]):
                    f.write(json.dumps({'src': l['token'],
                                        'tgt': label_to_tgt_list(l['label']) }) + '\n')

        else:
            with open(file.replace('.json', '_generated.json'), 'w') as f:
                for l in tqdm.tqdm(d[file]):
                    f.write(json.dumps({'src': l['token'],
                                        'tgt': label_to_tgt(l['label']) }) + '\n')

elif task == 'nyt':
    files = [
        './data/nyt/nyt_train_all.json',
        './data/nyt/nyt_val_all.json',
        './data/nyt/nyt_test_all.json']

    label_dict = {}
    hiera = defaultdict(set)
    with open('./data/nyt/nyt.taxonomy') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')

    class_label_dict = {}
    def loop_hiera(i, n):
        for j in list(hiera[i]):
            class_label_dict[j] = n
            if j in hiera:
                loop_hiera(j, n + 1)
    loop_hiera(-1, 0)


    d = {}
    label_sets = set()
    for i in files:
        d[i] = [json.loads(f) for f in open(i)]
        label_sets |= set(sum([j['label'] for j in d[i]], []))

    label_list = sorted(list(label_sets))

    assert len(label_list) == len(label_dict) == len(class_label_dict)

    label_map = {label: f'[A_{i}]' for i, label in enumerate(label_list)}

    import pickle
    with open('./data/nyt/label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    with open('./data/nyt/label_map.txt', 'w') as f:
        for i in label_map:
            f.write(f'{i}\t{label_map[i]}\n')

    def label_to_tgt(labels):
        index = [class_label_dict[label_dict[i]] for i in labels]

        labels = [label_map[i] for i in labels]

        ms = sorted(list(zip(index, labels)), key=lambda x: x[0])
        labels = [i[1] for i in ms]

        return ' '.join(labels)

    def label_to_tgt_list(labels):
        nyts = [[] for i in range(8)]

        index = [class_label_dict[label_dict[i]] for i in labels]
        labels = [label_map[i] for i in labels]
        for i, l in zip(index, labels):
            nyts[i].append(l)
        # print(''.join([str(len(i)) for i in nyts]))
        return nyts

    for file in d:
        assert '.json' in file
        if 'train' in file:
            with open(file.replace('.json', '_generated_tl.json'), 'w') as f:
                for l in d[file]:
                    f.write(json.dumps({'src': l['token'],
                                    'tgt': label_to_tgt_list(l['label']) }) + '\n')
        else:
            with open(file.replace('.json', '_generated.json'), 'w') as f:
                for l in d[file]:
                    f.write(json.dumps({'src': l['token'],
                                    'tgt': label_to_tgt(l['label']) }) + '\n')
