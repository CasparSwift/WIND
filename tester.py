import torch
import numpy as np
from model import loss_func


class Tester(object):
    def __init__(self, args, model, test_loader):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.loss_func = loss_func()

    def test_one_epoch(self, device, epoch):
        self.loss_func = self.loss_func.to(device)
        self.model.eval()
        correct_num, total = 0, 0
        pred_results = []
        for i, test_batch in enumerate(self.test_loader):
            ids, inputs = test_batch
            inputs = tuple(t.to(device) for t in inputs)
            input_ids = inputs[0]
            labels = inputs[2]
            logits = self.model(*inputs)[0]
            loss = self.loss_func(logits, labels)
            preds = np.argmax(logits.detach().cpu().numpy(), -1)
            preds = (preds == labels.cpu().numpy()).astype(np.int)
            correct_num += np.sum(preds)
            pred_results += preds.tolist()
            total += input_ids.shape[0]
            if i % self.args.logging_steps == 0:
                print(f'test loss: {loss.mean().item()}')            
        acc = 100 * correct_num / total
        # print(pred_results)
        return acc, pred_results


class TesterEnsemble(object):
    def __init__(self, args, model1, model2, test_loader):
        self.args = args
        self.model1 = model1
        self.model2 = model2
        self.test_loader = test_loader
        self.loss_func = loss_func()

    def test_one_epoch(self, device, epoch):
        self.loss_func = self.loss_func.to(device)
        self.model1.eval()
        self.model2.eval()
        correct_num, total = 0, 0
        for i, test_batch in enumerate(self.test_loader):
            ids, input_ids, masks, labels = tuple(t.to(device) for t in test_batch)
            logits1 = self.model1(input_ids, masks, labels)
            loss1 = self.loss_func(logits1, labels)
            logits2 = self.model2(input_ids, masks, labels)
            loss2 = self.loss_func(logits2, labels)
            loss = loss1 + loss2
            logits = torch.softmax(logits1, dim=-1) + torch.softmax(logits2, dim=-1)
            preds = np.argmax(logits.detach().cpu().numpy(), -1)
            correct_num += np.sum((preds == labels.cpu().numpy()).astype(np.int))
            total += input_ids.shape[0]
            if i % self.args.logging_steps == 0:
                print(f'test loss: {loss.mean().item()}')
        acc = 100 * correct_num / total
        print(f'ensemble acc: {acc}')
        return acc


class TesterDANN(object):
    def __init__(self, args, model, test_loader):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.loss_func = loss_func()

    def test_one_epoch(self, device, epoch):
        self.loss_func = self.loss_func.to(device)
        self.model.eval()
        correct_num, total = 0, 0
        pred_results = []
        for i, test_batch in enumerate(self.test_loader):
            ids, inputs = test_batch
            inputs = tuple(t.to(device) for t in inputs)
            input_ids = inputs[0]
            labels = inputs[2]
            logits = self.model(None, *inputs)[0]
            loss = self.loss_func(logits, labels)
            preds = np.argmax(logits.detach().cpu().numpy(), -1)
            preds = (preds == labels.cpu().numpy()).astype(np.int)
            correct_num += np.sum(preds)
            pred_results += preds.tolist()
            total += input_ids.shape[0]
            if i % self.args.logging_steps == 0:
                print(f'test loss: {loss.mean().item()}')
        acc = 100 * correct_num / total
        # print(pred_results)
        return acc, pred_results


TesterDict = {
    'meta': Tester,
    'baseline': Tester, 
    'meta_w': Tester, 
    'ensemble': TesterEnsemble, 
    'meta_batchw': Tester, 
    'DANN': TesterDANN,
}
