# baseline
# CUDA_VISIBLE_DEVICES='0' python main.py --batch_size=32 \
# 	--dataset=bdek --source=book --target=book --method=baseline --task=cls \
# 	--epoch_num=10 \

# meta-learning
CUDA_VISIBLE_DEVICES='0' python main.py --batch_size=32 --logging_step=100 \
	--val_batch_size=32 --test_batch_size=32 \
	--dataset=bdek --source=all --target=book --method=meta_w --task=cls \
	--alpha=0.15 --r=0.01 --epoch_num=10 --inner_step=1 --not_split \

# DANN
# CUDA_VISIBLE_DEVICES='' python main.py --batch_size=2 --logging_step=100 \
#     --val_batch_size=32 --test_batch_size=32 \
#     --dataset=bdek --source=all --target=book --method=DANN --task=cls \
#     --epoch_num=10 --not_split --gamma=0.5 \

# RE
# CUDA_VISIBLE_DEVICES='0' python main.py --batch_size=32 --test_batch_size=32 --val_batch_size=32 \
#   --method=meta_w --task=re --epoch_num=10 --logging_step=50 --alpha=0.1 --inner_step=2 \


# RE DANN
# CUDA_VISIBLE_DEVICES='0,1,2' python main.py --batch_size=32 --test_batch_size=32 --val_batch_size=32 \
#   --method=DANN --task=re --epoch_num=10 --logging_step=10 --alpha=0.1  --gamma=0.3 \

# fair_seq train
# train cross-domain
# train - train, valid - in-domain train, valid1 - tst2012
# test - tst2013, test1 - tst2014
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 \
# 	cd myfairseq && python train.py ../preprocessing/cross-domain-en-de-500K \
#     --task mytask --arch transformer_wmt_en_de --share-all-embeddings --max-sentences 50 \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4 \
#     --lr 0.0007 --stop-min-lr 1e-09 \
#     --criterion meta --label-smoothing 0.1 --weight-decay 0.0 \
#     --max-tokens 4096 --save-dir checkpoints/en-de-base \
#     --no-progress-bar --log-format json --log-interval 1 \
#     --save-interval-updates 10000 --keep-interval-updates 1 \
#     --train-subset train --valid-subset valid1 --max-update 20000

# preprocessing
# python build_domain_dataset.py --src en --tgt de --train_year 15 --test_year 16 \
# 	--dev tst2012 --test "tst2013 tst2014" \
# 	--out_of_domain wmt14_en_de_500K --output s --destdir cross-domain-en-de-500K \
# 	--binarize

