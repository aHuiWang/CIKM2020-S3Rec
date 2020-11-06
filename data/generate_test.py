# -*- coding: utf-8 -*-
# @Time    : 2020/11/6 15:51
# @Author  : Hui Wang

import numpy as np
from collections import defaultdict

np.random.seed(12345)

def sample_test_data(data_name, test_num=99, sample_type='random'):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.
    """

    data_file = f'{data_name}.txt'
    test_file = f'{data_name}_sample.txt'

    item_count = defaultdict(int)
    user_items = defaultdict()

    lines = open(data_file).readlines()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_items[user] = items
        for item in items:
            item_count[item] += 1

    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum([x for x in count])
    probability = [value / sum_value for value in count]

    user_neg_items = defaultdict()

    for user, user_seq in user_items.items():
        test_samples = []
        while len(test_samples) < test_num:
            if sample_type == 'random':
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else: # sample_type == 'pop':
                sample_ids = np.random.choice(all_item, test_num, replace=False, p=probability)
            sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in test_samples]
            test_samples.extend(sample_ids)
        test_samples = test_samples[:test_num]
        user_neg_items[user] = test_samples

    with open(test_file, 'w') as out:
        for user, samples in user_neg_items.items():
            out.write(user+' '+' '.join(samples)+'\n')

data_names = ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'Yelp', 'LastFM']
for data_name in data_names:
    sample_test_data(data_name)