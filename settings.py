import argparse
import os
import shutil


def parse_train_args():
    parser = argparse.ArgumentParser("Domain Adaptation by meta-learning")
    parser.add_argument("--debug", action="store_true")
    # meta: init w for each batch at training time
    # baseline: take all/in-domain data as source data
    # meta_w: init w for each instance at the begining of training time 
    parser.add_argument("--method", type=str, default='meta', 
        choices=['meta', 'baseline', 'meta_w', 'ensemble', 'meta_batchw', 'DANN'])
    parser.add_argument("--task", type=str, default='cls', 
        choices=['cls', 'nmt', 're'])
    parser.add_argument("--dataset", type=str, 
        choices=['bdek', 'small'])

    # model setting
    parser.add_argument("--model_dir", type=str, default="../bert-base-uncased")
    parser.add_argument("--hidden_size", type=int, default=768)
    
    # training and optimizer setting
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--early", type=float, default=1)

    # for meta
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--r", type=float, default=0.01)
    parser.add_argument("--not_split", action="store_true", default=False)

    # for DANN
    parser.add_argument("--gamma", type=float, default=1.0)

    # dataset setting
    parser.add_argument("--root", type=str, default='./data')
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--n_labels", type=int, default=2)
    parser.add_argument("--mod", type=int, default=1)
    parser.add_argument("--meta_ratio", type=float, default=0.5)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="model_ckpt")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--max_sent_length", type=int, default=256)
    

    args = parser.parse_args()

    # if args.debug:
    #     #args.n_train = 10000
    #     args.logging_steps = 1
    #     #args.n_test = 1000
    #     args.output_dir = "output_debug"
    #     args.overwrite = True
    #     args.ratio = 0.02

    # if os.path.exists(args.output_dir):
    #     if args.overwrite:
    #         choice = 'y'
    #     else:
    #         choice = input("Output directory ({}) exists! Remove? ".format(args.output_dir))
    #     if choice.lower()[0] == 'y':
    #         shutil.rmtree(args.output_dir)
    #         os.makedirs(args.output_dir)
    #     else:
    #         raise ValueError("Output directory exists!")
    # else:
    #     os.makedirs(args.output_dir)
    return args