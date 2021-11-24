from settings import parse_train_args
from model import BertForDA, DANN, BertForRE, DANNRE
from transformers import BertTokenizer, BertConfig
from trainers import MetaTrainer
from tester import TesterDict
from dataset import make_dataloader
import torch
import torch.nn as nn
import os

def setup_model(args, device):
    if args.task == 're':
        if args.method == 'DANN':
            model = DANNRE(args)
        else:
            model = BertForRE(args)
    elif args.task == 'cls':
        if args.method == 'DANN':
            model = DANN(args)
        else:
            model = BertForDA(args)
    else:
        exit()
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        print('no parallel')
    model = model.to(device)
    return model


def main():
    args = parse_train_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = setup_model(args, device)

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)

    results_dict = make_dataloader(args, tokenizer)
    train_loader, train_loader_val, valid_loader, test_loader = results_dict['loaders']
    train_num, in_domain_num = results_dict['train_num'], results_dict['in_domain_num']

    print(train_num, in_domain_num)
    trainer = MetaTrainer(args, model, train_loader, train_loader_val, train_num, in_domain_num)
    valid_tester = TesterDict[args.method](args, model, valid_loader)
    tester = TesterDict[args.method](args, model, test_loader)

    if args.method == 'ensemble':
        model2 = BertForDA(args)
        model_name1 = f'./{args.output_dir}/baseline_{args.target}-{args.target}.pt'
        model_name2 = f'./{args.output_dir}/baseline_out-{args.target}.pt'

        def remap(dict):
            return {k.replace('module.', ''): v for k, v in dict.items()}

        model.load_state_dict(remap(torch.load(model_name1, map_location=torch.device('cpu'))))
        model2.load_state_dict(remap(torch.load(model_name2, map_location=torch.device('cpu'))))
        tester = TesterEnsemble(args, model, model2, test_loader)
        tester.test_one_epoch(device, 0)
        return 0

    best_acc = 0 # best acc on test set
    best_valid_acc = 0 # best acc on valid set
    best_acc_select = 0 # best acc on selected model
    model_name = f'{args.task}_{args.method}_{args.source}-{args.target}.pt'
    if not os.path.exists('model_ckpt'):
        os.system('mkdir model_ckpt')
    model_dir = os.path.join('model_ckpt', model_name)
    for epoch in range(int(args.epoch_num * args.early)):
        print(f'Epoch: {epoch + 1}')
        if args.method == 'meta':
            trainer.train_one_epoch_meta(device, epoch + 1)
        elif args.method == 'meta_w':
            trainer.train_one_epoch_meta_w(device, epoch + 1)
        elif args.method == 'meta_batchw':
            trainer.train_one_epoch_meta_batchw(device, epoch + 1)
        elif args.method == 'DANN':
            trainer.train_one_epoch_DANN(device, epoch + 1)
        else:
            trainer.train_one_epoch(device, epoch + 1)

        # Test
        # valid_acc = valid_tester.test_one_epoch(device, epoch)
        # if epoch > 2:
        acc, pred_results = tester.test_one_epoch(device, epoch + 1)
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_dir)
            with open(model_dir.strip('.pt') + '-preds.txt', 'w') as f:
                f.write(str(pred_results))
        print(f'test_acc: {acc:.2f}, best_acc: {best_acc:.2f}')


if __name__ == '__main__':
    main()