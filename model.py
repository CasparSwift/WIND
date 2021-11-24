import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import BertModel, BertConfig


class cross_entropy_loss(nn.Module):
    def __init__(self):
        super(cross_entropy_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred, labels):
        loss = self.criterion(pred, labels)
        return loss


class weighted_cross_entropy_loss(nn.Module):
    def __init__(self):
        super(weighted_cross_entropy_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels, weight, soft):
        if soft:
            w = weight.sigmoid()
        else:
            w = (weight > 0)
        loss = (self.criterion(pred, labels) * w).sum() / w.sum()
        return loss


class loss_func(nn.Module):
    def __init__(self):
        super(loss_func, self).__init__()
        self.cross_entropy = cross_entropy_loss()
        self.weighted_cross_entropy = weighted_cross_entropy_loss()

    def forward(self, pred, labels, weight=None, soft=True):
        if weight is None:
            return self.cross_entropy(pred, labels)
        else:
            if weight.size(-1) == 1:
                return weight * self.cross_entropy(pred, labels)
            else:
                return self.weighted_cross_entropy(pred, labels, weight, soft)


class BertForDA(nn.Module):
    def __init__(self, args):
        super(BertForDA, self).__init__()
        print("Initializing main bert model...")
        self.bert_model = BertModel.from_pretrained(args.model_dir)
        self.classifier = torch.nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.n_labels)
        )

    def forward(self, input_ids, masks, labels):
        output = self.bert_model(input_ids, masks)[0]
        output = output[:, 0, :]
        logits = self.classifier(output)
        return logits, output


class GradReverse(Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class DANN(nn.Module):
    def __init__(self, args):
        super(DANN, self).__init__()
        print("Initializing main bert model...")
        self.bert_model = BertModel.from_pretrained(args.model_dir)
        self.labels_num = 2
        self.cc = torch.nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, self.labels_num)
        )
        self.dc = torch.nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 2)
        )

    def feature_extractor(self, tokens, masks):
        output = self.bert_model(tokens, masks)[0]
        return output[:, 0, :]

    def class_classifier(self, input):
        return self.cc(input)

    def domain_classifier(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        return self.dc(input)

    def forward(self, constant, *args):
        input_ids, masks, labels = args
        # feature of labeled data (source)
        feature_labeled = self.feature_extractor(input_ids, masks)
        # compute the class preds of src_feature
        class_preds = self.class_classifier(feature_labeled)
        # compute the domain preds of src_feature and target_feature
        labeled_preds = self.domain_classifier(feature_labeled, constant)
        return class_preds, labeled_preds


class NMTModel(nn.Module):
    pass


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BertForRE(nn.Module):
    def __init__(self, args):
        super(BertForRE, self).__init__()
        print("Initializing main bert model...")
        self.bert_model = BertModel.from_pretrained(args.model_dir)
        self.num_labels = 6
        self.cls_fc_layer = FCLayer(args.hidden_size, args.hidden_size, 0.5)
        self.entity_fc_layer = FCLayer(args.hidden_size, args.hidden_size, 0.5)
        self.label_classifier = FCLayer(
            args.hidden_size * 3,
            self.num_labels,
            0.5,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        # print(e_mask)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):
        outputs = self.bert_model(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # # Softmax
        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #     outputs = (loss,) + outputs

        # return outputs  # (loss), logits, (hidden_states), (attentions)
        return logits, concat_h


class DANNRE(nn.Module):
    def __init__(self, args):
        super(DANNRE, self).__init__()
        self.dc = nn.Linear(args.hidden_size * 3, 2)
        self.bert_for_re = BertForRE(args)

    def feature_extractor(self, tokens, masks):
        output = self.bert_model(tokens, masks)[0]
        return output[:, 0, :]

    def domain_classifier(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        return self.dc(input)

    def forward(self, constant, *args):
        # print(args)
        class_preds, h = self.bert_for_re(*args)
        labeled_preds = self.domain_classifier(h, constant)
        return class_preds, labeled_preds
