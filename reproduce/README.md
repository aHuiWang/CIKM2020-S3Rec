We release the pre-trained models for reproducibility.
```
{data-name}-epochs-{pretrain_epochs_num}.pt
```
The log files are also released and the '-0' means without pre-training.
```
Finetune_sample-{data-name}-epochs-0.pt
Finetune_sample-{data-name}-epochs-{pretrain_epochs_num}.pt
```
## Note

There is a minor bug in the old version codes, which is that we did not set random seed for all random methods.
 We are very sorry for this error. 
And we deleted pre-trained model considering the disk space. So we re-run the code and get new results, which could be
considered as reproduced the results in the paper.

When you fine-tune the model, please check the log information. If it is
```
ckp_file Not Found! The Model is same as SASRec.
```
then you actually run the SASRec and the model's parameters are initialized randomly. Otherwise you would see
```
Load Checkpoint From ckp_path!
```
which means you successfully initialize the model with pre-trained parameters.

### Meituan

Considering the **data security** issues, this dataset will not release.

### Beauty

| Model           | HR@1 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | MRR  |
|-----------------|------|------|--------|-------|---------|------|
| SASRec in paper |0.1870|0.3741|0.2848  |0.4696 |0.3156   |0.2852|
| SASRec in repo  |0.1867|0.3721|0.2843  |0.4651 |0.3142   |0.2850|
| S3-Rec in paper |0.2192|0.4502|0.3407  |0.5506 |0.3732   |0.3340|
| S3-Rec in repo  |0.2197|0.4626|0.3473  |0.5687 |0.3816   |0.3390|

+ pretrain (just use the default hyper-parameters)
```shell script
python run_pretrain.py \
--data_name Beauty
```

+ finetune (just use the default hyper-parameters)
```shell script
python run_finetune_sample.py \
--data_name Beauty \
--ckp 150
```

### Sports_and_Outdoors

| Model           | HR@1 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | MRR  |
|-----------------|------|------|--------|-------|---------|------|
| SASRec in paper |0.1455|0.3466|0.2497  |0.4622 |0.2869   |0.2520|
| SASRec in repo  |0.1472|0.3441|0.2487  |0.4645 |0.3875   |0.2524|
| S3-Rec in paper |0.1841|0.4267|0.3104  |0.5614 |0.3538   |0.3071|
| S3-Rec in repo  |0.1840|0.4319|0.3125  |0.5664 |0.3559   |0.3084|

+ pretrain (just use the default hyper-parameters)
```shell script
python run_pretrain.py \
--data_name Sports_and_Outdoors
```

+ finetune (just use the default hyper-parameters)
```shell script
python run_finetune_sample.py \
--data_name Sports_and_Outdoors \
--ckp 100
```

### Toys_and_Games

| Model           | HR@1 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | MRR  |
|-----------------|------|------|--------|-------|---------|------|
| SASRec in paper |0.1878|0.3682|0.2820  |0.4663 |0.3136   |0.2842|
| SASRec in repo  |0.1775|0.3683|0.2766  |0.4659 |0.3081   |0.2770|
| S3-Rec in paper |0.2003|0.4420|0.3270  |0.5530 |0.3629   |0.3202|
| S3-Rec in repo  |0.2070|0.4481|0.3335  |0.5593 |0.3695   |0.3268|

+ pretrain (just use the default hyper-parameters)
```shell script
python run_pretrain.py \
--data_name Toys_and_Games
```

+ finetune (just use the default hyper-parameters)
```shell script
python run_finetune_sample.py \
--data_name Toys_and_Games \
--ckp 150
```

### Yelp

| Model           | HR@1 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | MRR  |
|-----------------|------|------|--------|-------|---------|------|
| SASRec in paper |0.2375|0.5745|0.4113  |0.7373 |0.4642   |0.3927|
| SASRec in repo  |0.2310|0.5638|0.4017  |0.7384 |0.4582   |0.3856|
| S3-Rec in paper |0.2591|0.6085|0.4401  |0.7725 |0.4934   |0.4190|
| S3-Rec in repo  |0.2665|0.6195|0.4492  |0.7818 |0.5019   |0.4270|

+ pretrain (just use the default hyper-parameters)
```shell script
python run_pretrain.py \
--data_name Yelp
```

+ finetune (just use the default hyper-parameters)
```shell script
python run_finetune_sample.py \
--data_name Yelp \
--ckp 100
```

### LastFM

| Model           | HR@1 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | MRR  |
|-----------------|------|------|--------|-------|---------|------|
| SASRec in paper |0.1211|0.3385|0.2330  |0.4706 |0.2755   |0.2364|
| SASRec in repo  |0.1156|0.3092|0.2126  |0.4587 |0.2605   |0.2209|
| S3-Rec in paper |0.1743|0.4523|0.3156  |0.5835 |0.3583   |0.3072|
| S3-Rec in repo  |0.1569|0.4477|0.3056  |0.6083 |0.3577   |0.2981|

+ pretrain (just use the default hyper-parameters)
```shell script
python run_pretrain.py \
--data_name LastFM
```

+ finetune (just use the default hyper-parameters)
```shell script
python run_finetune_sample.py \
--data_name LastFM \
--ckp 150
```

