import random
import pickle
from collections import Counter

# for domain in ['beauty', 'book', 'electronics', 'music']:
#     idxs = list(range(6000))
#     random.shuffle(idxs)
#     train_idx = idxs[:1000]
#     valid_idx = idxs[1000:2000]
#     test_idx = idxs[2000:]
#     with open(f'./data/small/{domain}/idx.pkl', 'wb') as f:
#         pickle.dump((train_idx, valid_idx, test_idx), f)

for domain in ['book', 'dvd', 'electronics', 'kitchen']:
    idxs = list(range(6000))
    random.shuffle(idxs)
    train_idx = idxs[:1000]
    valid_idx = idxs[1000:2000]
    test_idx = idxs[2000:]
    with open(f'./data/amazon/{domain}/idx.pkl', 'wb') as f:
        pickle.dump((train_idx, valid_idx, test_idx), f)
