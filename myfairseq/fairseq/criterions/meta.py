# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import autograd
import random


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs if x is not None])


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def weighted_label_smoothed_nll_loss(lprobs, target, epsilon, repeat_w, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        # print(nll_loss.size())
        # print(smooth_loss.size())
        # print(nll_loss)
        w = repeat_w.sigmoid()
        # nll_loss = (nll_loss * w).sum()
        # print(w)
        # print(nll_loss.squeeze(-1))
        # print(w)
        nll_loss = (nll_loss.squeeze(-1) * w).sum()
        smooth_loss = (smooth_loss.squeeze(-1) * w).sum()
    eps_i = epsilon / lprobs.size(-1)
    # print(epsilon, eps_i, lprobs.size(-1))
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("meta")
class MetaCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def valid_forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def forward(self, model, sample, valid_sample=None, all_w=None, alpha=None, optimizer=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if valid_sample is None:
            return self.valid_forward(model, sample, reduce)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0007, eps=1e-8)
        # self.optimizer_pi = torch.optim.Adam(model.parameters(), lr=0.0007, eps=1e-8)
        self.optimizer_state_dict = optimizer.state_dict()

        self.theta = model.state_dict()
        batch_size = sample["target"].size(0)
        # w = torch.zeros(batch_size, requires_grad=True).to(sample["target"].device)
        sample_ids = sample['id']
        w = torch.index_select(all_w, dim=0, index=sample_ids.cpu()).to(sample["target"].device)
        w.requires_grad = True
        # print(w)

        # print(sample['target'].size())
        # print(sample['net_input']['src_tokens'].size())
        for step in range(1):
            # self.scheduler_pi.step()
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_weighted_loss(model, net_output, sample, w, reduce=reduce)
            # print(loss, nll_loss)
            with torch.autograd.profiler.record_function("pseudo backward"):
                optimizer.backward(loss)

            # valid data
            valid_net_output = model(**valid_sample["net_input"])
            valid_loss, valid_nll_loss = self.compute_loss(model, valid_net_output, valid_sample, reduce=reduce)
            # print(valid_loss, valid_nll_loss)
            # valid_loss_meter.update(valid_loss.item())
            grads_theta_hat = autograd.grad(valid_loss, model.parameters(), 
                allow_unused=True)
            grads_w = self.compute_grads_w(grads_theta_hat, sample, w, model, reduce)
            with torch.no_grad():
                w += alpha * grads_w
            # print(w, grads_w)
            del grads_theta_hat, loss, valid_loss, grads_w
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.load_state_dict(self.theta)
        w = w.detach()
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_weighted_loss(model, net_output, sample, w, reduce=reduce)
        # print(loss, nll_loss)
        if random.random() < 0.007:
            print(w.sigmoid())

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output, w

    def compute_grads_w(self, grads_theta_hat, sample, w, model, reduce):
        R = 0.01 / _concat(grads_theta_hat).norm()
        for p, v in zip(model.parameters(), grads_theta_hat):
            if v is not None:
                p.data.add_(R * v)

        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_weighted_loss(model, net_output, sample, w, reduce=reduce)
        grads1 = autograd.grad(loss, w)[0]

        for p, v in zip(model.parameters(), grads_theta_hat):
            if v is not None:
                p.data.sub_(2*R * v)

        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_weighted_loss(model, net_output, sample, w, reduce=reduce)
        grads2 = autograd.grad(loss, w)[0]

        for p, v in zip(model.parameters(), grads_theta_hat):
            if v is not None:
                p.data.add_(R * v)
        return (grads1 - grads2).div_(2*R)

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_weighted_loss(self, model, net_output, sample, w, reduce=True):
        # log probs [batch_size * sent length, vocab length]
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # print(w)
        # print(w.size())
        # print(lprobs.size())
        # target [batch_size * sent length, 1]
        # print(target)
        bsz = int(w.size(0))
        sent_len = int(lprobs.size(0)) // bsz
        repeat_w = torch.unsqueeze(w, -1).expand(bsz, sent_len)
        repeat_w = repeat_w.contiguous().view(-1)
        # print(repeat_w)
        # print(repeat_w.size())
        loss, nll_loss = weighted_label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            repeat_w,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
