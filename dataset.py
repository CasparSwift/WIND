from torch.utils.data import Dataset, random_split, DataLoader
from multiprocessing import Pool
import pickle
import random
import torch
import os
import json



def dynamic_collate_fn(batch):
    idx, input_ids, labels = list(zip(*batch))
    max_len = max(len(inp) for inp in input_ids)
    masks = [[1]*len(inp) + [0]*(max_len-len(inp)) for inp in input_ids]
    input_ids = [inp + [0]*(max_len-len(inp)) for inp in input_ids]
    return_vals = (input_ids, masks, labels)
    return torch.tensor(idx, dtype=torch.long), tuple(torch.tensor(t, dtype=torch.long) for t in return_vals)


def dynamic_collate_fn_re(batch):
    idx, input_ids, e_mask1, e_mask2, labels = list(zip(*batch))
    max_len = max(len(inp) for inp in input_ids)
    masks = [[1]*len(inp) + [0]*(max_len-len(inp)) for inp in input_ids]
    input_ids = [inp + [0]*(max_len-len(inp)) for inp in input_ids]
    e_mask1 = [m + [0]*(max_len-len(m)) for m in e_mask1] 
    e_mask2 = [m + [0]*(max_len-len(m)) for m in e_mask2] 
    return_vals = (input_ids, masks, labels, e_mask1, e_mask2)
    return torch.tensor(idx, dtype=torch.long), tuple(torch.tensor(t, dtype=torch.long) for t in return_vals)


class DataReader(object):
    def __init__(self, args, tokenizer, split, ratio=0.5):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.root = args.root
        self.data_val = None

        if split == "train":
            if args.method == 'baseline':
                # only in-domain training data
                if args.source == args.target:
                    self.data = [data for i, data in enumerate(self.read_data(domain=args.target, split="train")) if i % args.mod == 0]
                # only out-of-domain data
                elif args.source == 'out':
                    self.data = self.get_out_of_domain()
                # all data
                elif args.source == 'all':
                    self.data = [data for i, data in enumerate(self.read_data(domain=args.target, split="train")) if i % args.mod == 0] \
                        + self.get_out_of_domain()
                else:
                    exit()
                self.data = self.encode_data(self.data)
            else:
                self.data = [data for i, data in enumerate(self.read_data(domain=args.target, split="train")) if i % args.mod == 0]
                size = int(ratio * len(self.data))
                random.shuffle(self.data)
                if args.not_split:
                    import copy
                    self.data_val = copy.deepcopy(self.data)
                else:
                    self.data, self.data_val = self.data[:size], self.data[size:]
                self.data += self.get_out_of_domain()
                self.data = self.encode_data(self.data)
                self.data_val = self.encode_data(self.data_val)
        else:
            self.data = self.read_data(domain=args.target, split=split)
            self.data = self.encode_data(self.data)

    def get_dataset(self):
        dataset_class = dataset_dict[self.args.dataset]
        if self.data_val is None:
            return dataset_class(self.data)
        else:
            return dataset_class(self.data), dataset_class(self.data_val)

    # get out-of-domain data
    def get_out_of_domain(self):
        s = []
        for domain in dataset_dict[self.args.dataset].domains():
            if domain != self.args.target:
                s += self.read_data(domain=domain, split="all")
        return s

    def read_data(self, domain, split):
        data, labels = [], []
        for filename in ['neg', 'un_neg', 'pos', 'un_pos']:
            with open(os.path.join(self.root, f'amazon/{domain}/{filename}.txt'), 'r') as f:
                lines = f.read().strip('\n').split('\n')
                labels += [0 if 'neg' in filename else 1 for _ in range(len(lines))]
                data += lines
        assert len(data) == len(labels)
        results = list(zip(data, labels))
        with open(os.path.join(self.root, f'amazon/{domain}/idx.pkl'), 'rb') as f:
            if split == 'train':
                iidx = pickle.load(f)[0]
            elif split == 'valid':
                iidx = pickle.load(f)[1]
            elif split == 'test':
                iidx = pickle.load(f)[2]
            elif split == 'all':
                return results
            else:
                exit()
            return [results[i] for i in iidx]

    def map_data(self, data_and_labels):
        data, label = data_and_labels
        input_ids = self.tokenizer.encode(data, 
            max_length=self.args.max_sent_length, add_special_tokens=True, truncation=True)
        return input_ids, label

    def encode_data(self, d):
        with Pool(self.args.n_workers) as pool:
            d = pool.map(self.map_data, d)
        d = [(i,) + j for i, j in enumerate(d)]
        # random.shuffle(self.data)
        return d


# TODO
# class Small_Amazon_Dataset(Dataset):
#     def __init__(self, args, tokenizer, split):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.split = split
#         self.root = args.root
#         self.domains = ['beauty', 'book', 'electronics', 'music']
#         self.source_domain = args.source
#         self.target_domain = args.target
#         self.label_map = {1:0, 2:0, 3:1, 4:2, 5:2}

#         self.data = []
#         if split == 'train':
#             if self.source_domain != self.target_domain:
#                 for domain in self.domains:
#                     if domain != self.target_domain:
#                         self.data += self.read_data(domain, idx=False)
#             if self.source_domain != 'out':
#                 self.data += self.read_data(self.target_domain, idx=True)
#         elif split == 'valid':
#             self.data += self.read_data(self.target_domain, idx=True)
#         elif split == 'test':
#             self.data += self.read_data(self.target_domain, idx=True)
#         else:
#             exit()
#         # random.shuffle(self.data)
#         with Pool(args.n_workers) as pool:
#             self.data = pool.map(self.map_data, self.data)
#         self.data = [(i,) + d for i, d in enumerate(self.data)]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

#     def read_data(self, domain):
#         with open(os.path.join(self.root, f'small/{domain}/set1_text.txt'), 'r') as f:
#             data = f.read().strip('\n').split('\n')
#         with open(os.path.join(self.root, f'small/{domain}/set1_label.txt'), 'r') as f:
#             labels = f.read().strip('\n').split('\n')
#         assert len(data) == len(labels)
#         results = list(zip(data, labels))
#         with open(os.path.join(self.root, f'small/{domain}/idx.pkl'), 'rb') as f:
#             if self.split == 'train':
#                 iidx = pickle.load(f)[0]
#             elif self.split == 'valid':
#                 iidx = pickle.load(f)[1]
#             elif self.split == 'test':
#                 iidx = pickle.load(f)[2]
#             elif self.split == 'all':
#                 return results
#             else:
#                 exit()
#             return [results[i] for i in iidx]

#     def map_data(self, data_and_labels):
#         data, label = data_and_labels
#         input_ids = self.tokenizer.encode(data, 
#             max_length=self.args.max_sent_length, add_special_tokens=True, truncation=True)
#         # for l in labels:
#         #     print(l)
#         #     print(l.split('.'))
#         label = self.label_map[int(label.split('.')[0])]
#         return input_ids, label


class Amazon_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 

    @staticmethod
    def domains():
        return ['book', 'dvd', 'electronics', 'kitchen']


class RE_Dataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.tokenizer = tokenizer
        self.args = args
        if split == 'train': # in+out
            with open(os.path.join(args.root, f'ace2005/in-domain.json'), 'r') as f:
                self.json_data = json.load(f)
            with open(os.path.join(args.root, f'ace2005/out-of-domain.json'), 'r') as f:
                self.json_data += json.load(f)
        else:
            with open(os.path.join(args.root, f'ace2005/{split}.json'), 'r') as f:
                self.json_data = json.load(f)
        symbol1 = self.tokenizer.encode('$', add_special_tokens=False)[0]
        symbol2 = self.tokenizer.encode('#', add_special_tokens=False)[0]
        self.data = []
        self.label_map = {'GEN-AFF':0, 'ORG-AFF':1, 'PHYS':2, 
                          'PART-WHOLE':3, 'ART':4, 'PER-SOC':5}
        for data in self.json_data:
            for s1, t1, s2, t2, relation in data['relations']:
                if s1 > s2:
                    s1, t1, s2, t2 = s2, t2, s1, t1
                try:
                    assert s1 < t1 and t1 <= s2 and s2 < t2, f"{s1} {t1} {s2} {t2} {data}"
                except:
                    continue
                s = data['tokens']
                tokens1 = self.encode(['[CLS]'] + s[:s1])
                tokens2 = self.encode(['$'] + s[s1:t1] + ['$'])
                tokens3 = self.encode(s[t1:s2])
                tokens4 = self.encode(['#'] + s[s2:t2] + ['#'])
                tokens5 = self.encode(s[t2:])
                input_ids = tokens1 + tokens2 + tokens3 + tokens4 + tokens5
                e_mask1 = [0] * len(tokens1) + [1] * len(tokens2) + [0] * (len(tokens3+tokens4+tokens5))
                e_mask2 = [0] * len(tokens1+tokens2+tokens3) + [1] * len(tokens4) + [0] * len(tokens5)
                # for item in zip(input_ids, e_mask1, e_mask2):
                #     print(item)
                # print('-'*10)
                self.data.append((input_ids, e_mask1, e_mask2, self.label_map[relation]))
        self.data = [(i,) + j for i, j in enumerate(self.data)]
        if split == 'train':
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def encode(self, tokens):
        return self.tokenizer.encode(' '.join(tokens), 
            max_length=self.args.max_sent_length, add_special_tokens=False, truncation=True)


def make_dataloader(args, tokenizer):
    if args.task == 're':
        train_set = RE_Dataset(args, tokenizer, "train")
        indom_set = RE_Dataset(args, tokenizer, "in-domain")
        out_set = RE_Dataset(args, tokenizer, "out-of-domain")
        valid_set = RE_Dataset(args, tokenizer, "dev")
        test_set = RE_Dataset(args, tokenizer, "test")
        valid_loader = DataLoader(valid_set, batch_size=args.val_batch_size, shuffle=False, 
            num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn_re)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, 
            num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn_re)
        if args.method == 'baseline':
            train_loader = DataLoader(indom_set, batch_size=args.batch_size, shuffle=True, 
                num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn_re)
            train_loader_val = None
            print(f'train: {len(train_set)}, valid: {len(valid_set)}, test: {len(test_set)}')
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn_re)
            train_loader_val = DataLoader(indom_set, batch_size=args.batch_size, shuffle=True, 
                num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn_re)
            print(f'train: {len(train_set)} and {len(indom_set)}, valid: {len(valid_set)}, test: {len(test_set)}')
        return {'loaders': (train_loader, train_loader_val, valid_loader, test_loader),
            'train_num': len(train_set), 
            'in_domain_num': len(indom_set)}

    dataset_class = dataset_dict[args.dataset]
    valid_set = DataReader(args, tokenizer, "valid").get_dataset()
    test_set = DataReader(args, tokenizer, "test").get_dataset()
    valid_loader = DataLoader(valid_set, batch_size=args.val_batch_size, shuffle=False, 
        num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, 
        num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn)

    if args.method == 'baseline':
        train_set = DataReader(args, tokenizer, "train").get_dataset()
        train_set_val = []
        print(f'train: {len(train_set)}, valid: {len(valid_set)}, test: {len(test_set)}')

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.n_workers, drop_last=True, collate_fn=dynamic_collate_fn)
        train_loader_val = None
    else:
        train_set, train_set_val = DataReader(args, tokenizer, "train", args.meta_ratio).get_dataset()
        print(f'train: {len(train_set)} and {len(train_set_val)}, valid: {len(valid_set)}, test: {len(test_set)}')

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn)
        train_loader_val = DataLoader(train_set_val, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.n_workers, drop_last=False, collate_fn=dynamic_collate_fn)
    return {'loaders': (train_loader, train_loader_val, valid_loader, test_loader),
            'train_num': len(train_set), 
            'in_domain_num': int(args.meta_ratio/(1-args.meta_ratio)*len(train_set_val))}


dataset_dict = {
    # 'small': Small_Amazon_Dataset,
    'bdek': Amazon_Dataset
}
