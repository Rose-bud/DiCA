# DiCA
The code of DiCA: Disambiguated Contrastive Alignment for Cross-Modal Retrieval with Partial Labels（AAAI2025）

#
CUDA_VISIBLE_DEVICES=1 python DiCA.py --train_batch_size 32 --eval_batch_size 256 --partial_ratio 0.2 --data_name wiki --log_name {your_log_name} --partial_file {your_partial_file} --lr 0.0001 --num_class 10 --max_epochs 100 --w1 1 --w2 2.5 --w3 2.5 --output_dim 10 --lamda 5 --method ours

#
e.g. For Wikipedia with the partial rate of 0.1
#

CUDA_VISIBLE_DEVICES=1 python DiCA.py --train_batch_size 32 --eval_batch_size 128 --partial_ratio 0.1 --data_name wiki --log_name partiallabel_ours_wiki_0.1 --partial_file 'wiki/partial_labels_0.1_sym.json' --lr 0.0001 --num_class 10 --max_epochs 100 --w1 1 --w2 2.5 --w3 2.5 --output_dim 10 --lamda 5 --method ours

# All the datasets
https://pan.baidu.com/s/1nbZlGXFDKjxhZyC2qs0YIA?pwd=DiCA
