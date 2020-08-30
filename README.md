
Code for our CIKM 2020 Paper ["**S3-Rec: Self-Supervised Learning for Sequential
 Recommendation with Mutual Information Maximization"**](https://arxiv.org/pdf/2008.07873.pdf)

# Overview
![avatar](model.PNG)
# NOTE
In the PAPER, we pair the ground-truth item 
with 99 randomly sampled negative items that the user
has not interacted with, and report the results of 
HR@{1, 5, 10}, NDCG@{5, 10} and MRR.

In the repo, we rank the ground-truth item with ALL the items.
We omit the FM and AutoInt because they need 
enumerate all user-item pairs, which take a very long time. 
The results are shown in the following pic(all_rank.PNG).

Performance comparison of different methods on six datasets. The best performance and the second best performance
methods are denoted in bold and underlined fonts respectively.

![avatar](all_rank.PNG)

### requirements
```shell script
pip install -r requirements.txt
```

### data preprocess
```shell script
./data/data_process.py

data-name.txt
one user per line
user_1 item_1 item_2 ...
user_2 item_1 item_2 ...

data-name_item2attributes.json
{item_1:[attr, ...], item_2:[attr, ...], ... }
```

### pretrain
```shell script
python run_pretrain.py \
--data_name Beauty
```

### finetune
```shell script
python run_finetune.py \
--data_name Beauty \
--ckp 100
```