import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np
from utils import *

def get_loss_model(cfg):
        return nn.CrossEntropyLoss().cuda()

def cross_entropy_with_multi_targets(input_probs, targets):
    loss = 0.
    count = 0
    for i in input_probs:
        for t in targets:
            loss += torch.log(torch.sum(torch.exp(i))) - i[t]
            count += 1
    loss = loss / count
    return loss

def get_optimizer(cfg, params):
    if cfg["optimizer"]["type"] == "adam":
        optimizer = torch.optim.Adam(params,
                        lr=cfg["optimizer"]["lr"],
                        weight_decay=cfg["optimizer"]["weight_decay"])
    elif cfg["optimizer"]["type"] == "sgd":
        optimizer = torch.optim.SGD(params,
                        lr=cfg["optimizer"]["lr"],
                        weight_decay=cfg["optimizer"]["weight_decay"],
                        momentum=cfg["optimizer"]["momentum"])
    return optimizer

def get_lr_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer,
                        step_size=cfg["optimizer"]["step_size"], gamma=cfg["optimizer"]["gamma"])

class CarColor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_data = cfg["data"]

        self.resnet50 = resnet50(pretrained=True)
        num_features = self.resnet50.fc.in_features
        num_class = getTotalColorLabel()
        self.resnet50.fc = nn.Linear(num_features, num_class) # length of color classes
        self.loss_model = get_loss_model(self.cfg)

        if self.cfg["train"]["loss_calculator"] == "cross_entropy_by_one_hot_enc":
            self.compute_loss = self.compute_loss_by_one_hot_enc

        elif self.cfg["train"]["loss_calculator"] == "cross_entropy_by_ans_prob_enc":
            self.compute_loss = self.compute_loss_by_ans_prob_enc

    def forward(self, track):
        return self.resnet50(track)

    def compute_loss_by_one_hot_enc(self, track):
        loss = 0.
        for i, t in enumerate(track['crops']):
            l = 0.
            out = self.forward(t)

            for k, v in track['color'][i].items():
                target = torch.LongTensor([k] * len(out)).cuda()
                l += self.loss_model(out, target) * v
            loss += l
        loss /= len(track['crops'])
        return loss

    def compute_loss_by_ans_prob_enc(self, track):
        loss = 0.
        for i, t in enumerate(track['crops']):
            l = 0.
            out = self.forward(t)
            pred = F.log_softmax(out, dim=-1)
            t_pred = torch.transpose(pred, 0, 1)
            for k, v in track['color'][i].items():
                l += (t_pred[k] * -1 * v).mean()
            loss += l
        loss /= len(track['crops'])
        return loss

    def compute_color_list(self, query):
        return getColorList(query)

    def compute_similarity_on_frame(self, tracks, colors):
        with torch.no_grad():
            scores = list()
            for i, t in enumerate(tracks['crops']):
                score = 0.
                out = self.forward(t)
                actual_out = out[:tracks['sequence_len'][i]]
                #_, pre = torch.max(actual_out, 1)
                percentage = nn.functional.softmax(actual_out, dim=1)[0]
                percentage = percentage.detach().to('cpu').numpy()

                for c in colors:
                    score += percentage[c]
                scores.append(score)
        return scores

class CarType(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_data = cfg["data"]

        self.resnet50 = resnet50(pretrained=True)
        num_features = self.resnet50.fc.in_features
        num_class = getTotalTypeLabel()
        self.resnet50.fc = nn.Linear(num_features, num_class) # length of model classes
        self.loss_model = get_loss_model(self.cfg)

        if self.cfg["train"]["loss_calculator"] == "cross_entropy_by_one_hot_enc":
            self.compute_loss = self.compute_loss_by_one_hot_enc

        elif self.cfg["train"]["loss_calculator"] == "cross_entropy_by_ans_prob_enc":
            self.compute_loss = self.compute_loss_by_ans_prob_enc

    def forward(self, track):
        return self.resnet50(track)

    def compute_loss_by_one_hot_enc(self, track):
        loss = 0.
        for i, t in enumerate(track['crops']):
            l = 0.
            out = self.forward(t)

            for k, v in track['type'][i].items():
                target = torch.LongTensor([k] * len(out)).cuda()
                l += self.loss_model(out, target) * v
            loss += l
        loss /= len(track['crops'])
        return loss

    def compute_loss_by_ans_prob_enc(self, track):
        loss = 0.
        for i, t in enumerate(track['crops']):
            l = 0.
            out = self.forward(t)
            pred = F.log_softmax(out, dim=-1)
            t_pred = torch.transpose(pred, 0, 1)
            for k, v in track['type'][i].items():
                l += (t_pred[k] * -1 * v).mean()
            loss += l
        loss /= len(track['crops'])
        return loss

    def compute_type_list(self, query):
        return getTypeList(query)

    def compute_similarity_on_frame(self, tracks, types):
        with torch.no_grad():
            scores = list()
            for i, t in enumerate(tracks['crops']):
                score = 0.
                out = self.forward(t)
                actual_out = out[:tracks['sequence_len'][i]]
                #_, pre = torch.max(actual_out, 1)
                percentage = nn.functional.softmax(actual_out, dim=1)[0]
                percentage = percentage.detach().to('cpu').numpy()

                for tp in types:
                    score += percentage[tp]
                scores.append(score)
        return scores



