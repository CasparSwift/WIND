import torch
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from copy import deepcopy
from model import loss_func
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs if x is not None])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class MetaTrainer(object):
    def __init__(self, args, model, train_loader, valid_loader, train_num, in_domain_num):
        self.args = args
        self.model = model
        # self.model_pi = deepcopy(model)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_num = train_num
        self.in_domain_num = in_domain_num

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}')
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=args.warmup*total_steps,
                                                         num_training_steps=total_steps)
        self.optimizer_pi = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        self.scheduler_pi = get_linear_schedule_with_warmup(self.optimizer_pi,
                                                            num_warmup_steps=args.warmup*total_steps,
                                                            num_training_steps=total_steps)
        self.alpha = args.alpha
        self.r = args.r
        self.loss_func = loss_func()

        if args.method == 'meta_w':
            self.w = torch.zeros(self.train_num, requires_grad=False)
            # weight of each in-domain instance will be assigned a large number
            for i in range(self.in_domain_num):
                self.w[i] = 1e8

    def compute_grads_w(self, grads_theta_hat, w, *inputs):
        R = self.r / _concat(grads_theta_hat).norm()
        y_train = inputs[2]
        for p, v in zip(self.model.parameters(), grads_theta_hat):
            if v is not None:
                p.data.add_(R * v)

        logits = self.model(*inputs)[0]
        loss = self.loss_func(logits, y_train, w)
        grads1 = autograd.grad(loss, w)[0]

        for p, v in zip(self.model.parameters(), grads_theta_hat):
            if v is not None:
                p.data.sub_(2*R * v)

        logits = self.model(*inputs)[0]
        loss = self.loss_func(logits, y_train, w)
        grads2 = autograd.grad(loss, w)[0]

        for p, v in zip(self.model.parameters(), grads_theta_hat):
            if v is not None:
                p.data.add_(R * v)
        return (grads1 - grads2).div_(2*R)

    def train_one_epoch(self, device, epoch):
        loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, train_batch in enumerate(self.train_loader):
            # print(train_batch)
            self.scheduler.step()
            # train batch
            ids, inputs = train_batch
            inputs = tuple(t.to(device) for t in inputs)
            labels = inputs[2]
            logits = self.model(*inputs)[0]
            loss = self.loss_func(logits, labels)
            loss_meter.update(loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) '\
                    f'| lr: {self.scheduler.get_lr()[0]:.8f} ')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_one_epoch_meta(self, device, epoch):
        loss_meter = AverageMeter()
        valid_loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, train_batch in enumerate(self.train_loader):
            self.scheduler.step()
            # train batch
            ids, input_ids, masks, labels = tuple(t.to(device) for t in train_batch)
            # valid batch
            val_batch = next(iter(self.valid_loader))
            ids_val, input_ids_val, masks_val, labels_val = tuple(t.to(device) for t in val_batch)
            batch_size = input_ids.shape[0]

            # save real optimizer state
            self.optimizer.zero_grad()
            self.optimizer_state_dict = self.optimizer.state_dict()
            self.optimizer_pi.load_state_dict(self.optimizer_state_dict)

            self.theta = self.model.state_dict()

            w = torch.zeros(batch_size, requires_grad=True).to(device)
            # optimizer_w = torch.optim.SGD([w], 0.1)
            for step in range(self.args.inner_steps):
                logits = self.model(input_ids, masks, labels)
                loss = self.loss_func(logits, labels, w, soft=True)
                # grad_theta = autograd.grad(loss, self.model_pi.parameters(), 
                #     create_graph=True, allow_unused=True)
                self.optimizer_pi.zero_grad()
                loss.backward()
                self.optimizer_pi.step()
                logits = self.model(input_ids_val, masks_val, labels_val)
                valid_loss = self.loss_func(logits, labels_val)
                valid_loss_meter.update(valid_loss.item())
                grads_theta_hat = autograd.grad(valid_loss, self.model.parameters(), 
                    allow_unused=True)
                grads_w = self.compute_grads_w(grads_theta_hat, input_ids, masks, labels, w)
                with torch.no_grad():
                    w += self.alpha * grads_w
                # print(w, grads_w)
                del grads_theta_hat, loss, valid_loss, grads_w

            # w = 100 * (torch.tensor(w.sigmoid() > 0.5, dtype=torch.float) - 0.5).to(device)
            self.model.load_state_dict(self.theta)
            self.optimizer.load_state_dict(self.optimizer_state_dict)

            logits = self.model(input_ids, masks, labels)
            loss = self.loss_func(logits, labels, w, soft=True)
            loss_meter.update(loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| train loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) '\
                    f'| valid loss: {valid_loss_meter.val:.4f}({valid_loss_meter.avg:.4f})')
                print(w.sigmoid())
            loss.backward()
            self.optimizer.step()

    def train_one_epoch_meta_w(self, device, epoch):
        loss_meter = AverageMeter()
        valid_loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, train_batch in enumerate(self.train_loader):
            self.scheduler.step()
            # train batch
            ids, inputs = train_batch
            inputs = tuple(t.to(device) for t in inputs)
            labels = inputs[2]
            # valid batch
            val_batch = next(iter(self.valid_loader))
            ids_val, inputs_val = val_batch
            inputs_val = tuple(t.to(device) for t in inputs_val)
            labels_val = inputs_val[2]
            # batch_size = input_ids.shape[0]

            # save real optimizer state
            
            # self.optimizer_state_dict = self.optimizer.state_dict()

            self.theta = self.model.state_dict()
            w = torch.index_select(self.w, dim=0, index=ids.cpu()).to(device)
            w.requires_grad = True
            for step in range(self.args.inner_steps):
                # if step == 0:
                self.scheduler_pi.step()
                logits = self.model(*inputs)[0]
                loss = self.loss_func(logits, labels, w, soft=True)
                # grad_theta = autograd.grad(loss, self.model_pi.parameters(), 
                #     create_graph=True, allow_unused=True)
                # self.optimizer_pi.load_state_dict(self.optimizer_state_dict)
                self.optimizer_pi.zero_grad()
                loss.backward()
                self.optimizer_pi.step()
                logits = self.model(*inputs_val)[0]
                valid_loss = self.loss_func(logits, labels_val)
                valid_loss_meter.update(valid_loss.item())
                grads_theta_hat = autograd.grad(valid_loss, self.model.parameters(), 
                    allow_unused=True)
                grads_w = self.compute_grads_w(grads_theta_hat, w, *inputs)
                with torch.no_grad():
                    w += self.alpha * grads_w
                    # print(w, grads_w)
                del grads_theta_hat, loss, valid_loss, grads_w
            self.model.load_state_dict(self.theta)
            # self.optimizer.load_state_dict(self.optimizer_state_dict)

            w = w.detach()
            logits = self.model(*inputs)[0]
            loss = self.loss_func(logits, labels, w, soft=True)
            loss_meter.update(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save the optimized w
            w = w.cpu()
            for j, id in enumerate(ids.tolist()):
                # update the weight of the out-of-domain data only
                if id >= self.in_domain_num:
                    self.w[id] = w[j]
            
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| train loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) '\
                    f'| valid loss: {valid_loss_meter.val:.4f}({valid_loss_meter.avg:.4f})')
                # print(w.sigmoid())
                # print('self.w: ', self.w)            

    def train_one_epoch_meta_batchw(self, device, epoch):
        loss_meter = AverageMeter()
        valid_loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, train_batch in enumerate(self.train_loader):
            self.scheduler.step()
            # train batch
            ids, input_ids, masks, labels = tuple(t.to(device) for t in train_batch)
            # valid batch
            val_batch = next(iter(self.valid_loader))
            ids_val, input_ids_val, masks_val, labels_val = tuple(t.to(device) for t in val_batch)
            batch_size = input_ids.shape[0]

            # save real optimizer state
            self.optimizer.zero_grad()
            self.optimizer_state_dict = self.optimizer.state_dict()
            self.optimizer_pi.load_state_dict(self.optimizer_state_dict)

            self.theta = self.model.state_dict()

            w = torch.tensor([1.0], requires_grad=True).to(device)
            for step in range(self.args.inner_steps):
                logits = self.model(input_ids, masks, labels)
                loss = self.loss_func(logits, labels, w, soft=True)
                # grad_theta = autograd.grad(loss, self.model_pi.parameters(), 
                #     create_graph=True, allow_unused=True)
                self.optimizer_pi.zero_grad()
                loss.backward()
                self.optimizer_pi.step()
                logits = self.model(input_ids_val, masks_val, labels_val)
                valid_loss = self.loss_func(logits, labels_val)
                valid_loss_meter.update(valid_loss.item())
                grads_theta_hat = autograd.grad(valid_loss, self.model.parameters(), 
                    allow_unused=True)
                grads_w = self.compute_grads_w(grads_theta_hat, input_ids, masks, labels, w)
                with torch.no_grad():
                    w += self.alpha * grads_w
                # print(w, grads_w)
                del grads_theta_hat, loss, valid_loss, grads_w

            self.model.load_state_dict(self.theta)
            self.optimizer.load_state_dict(self.optimizer_state_dict)

            w = w.detach()
            logits = self.model(input_ids, masks, labels)
            loss = self.loss_func(logits, labels, w, soft=True)
            loss_meter.update(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| train loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) '\
                    f'| valid loss: {valid_loss_meter.val:.4f}({valid_loss_meter.avg:.4f})')

    def train_one_epoch_DANN(self, device, epoch):
        start_steps = (epoch - 1) * len(self.train_loader)
        total_steps = self.args.epoch_num * len(self.train_loader)
        loss_meter = AverageMeter()
        cls_meter = AverageMeter()
        domloss_meter = AverageMeter()
        acc_meter = AverageMeter()
        dom_acc_meter = AverageMeter()
        time_meter = AverageMeter()
        self.model.train()
        for i, train_batch in enumerate(self.train_loader):
            self.scheduler.step()
            # train batch
            # ids, input_ids, masks, labels = tuple(t.to(device) for t in train_batch)
            ids, inputs = train_batch
            inputs = tuple(t.to(device) for t in inputs)
            labels = inputs[2]
            dom_labels = torch.tensor(ids < self.in_domain_num, dtype=torch.long).to(device)

            # setup hyperparameters
            p = float(i + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-self.args.gamma * p)) - 1

            # forward
            class_preds, dom_preds = self.model(constant, *inputs)
            cls_loss = self.loss_func(class_preds, labels)
            domain_loss = self.loss_func(dom_preds, dom_labels)
            loss = cls_loss + domain_loss
            
            preds = np.argmax(dom_preds.detach().cpu().numpy(), -1)
            correct_num = np.sum((preds == dom_labels.cpu().numpy()).astype(np.int))
            dom_acc = correct_num / preds.shape[0]

            loss_meter.update(float(loss))
            cls_meter.update(float(cls_loss))
            domloss_meter.update(float(domain_loss))
            # acc_meter.update(float(acc))
            dom_acc_meter.update(float(dom_acc))

            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {cls_meter.val:.4f}({cls_meter.avg:.4f}) |'\
                    f' {dom_acc_meter.val:.4f}({dom_acc_meter.avg:.4f}) '\
                    f'| lr: {self.scheduler.get_lr()[0]:.8f} ')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            