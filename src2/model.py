import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
from utils import *


def get_loss_model(cfg):
    return nn.CrossEntropyLoss().cuda()

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
        '''
        state_dict = torch.load(self.model_cfg.RESNET_CHECKPOINT,
                                map_location=lambda storage, loc: storage.cpu())
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        self.resnet50.load_state_dict(state_dict, strict=False)
        '''

    def forward(self, track):
        return self.resnet50(track)

    def compute_loss(self, track):
        loss = 0.
        out = self.forward(track['crop'])

        for index, o in enumerate(out):
            o = torch.unsqueeze(o, dim=0)
            l = 0.
            for k, v in track['color'][index].items():
                target = torch.LongTensor([k]).cuda()
                l += self.loss_model(o, target) * v
            loss += l
        loss /= len(track['crop'])
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
        '''
        state_dict = torch.load(self.model_cfg.RESNET_CHECKPOINT,
                                map_location=lambda storage, loc: storage.cpu())
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        self.resnet50.load_state_dict(state_dict, strict=False)
        '''
    def forward(self, track):
        return self.resnet50(track)

    def compute_loss(self, track):  # need to check if it's correct !!!!!!!!!!!!!!!!!!!
        loss = 0.
        out = self.forward(track['crop'])

        for index, o in enumerate(out):
            o = torch.unsqueeze(o, dim=0)
            l = 0.
            for k, v in track['type'][index].items():
                target = torch.LongTensor([k]).cuda()
                l += self.loss_model(o, target) * v
            loss += l
        loss /= len(track['crop'])
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



