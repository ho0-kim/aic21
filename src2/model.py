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

    def compute_color_embeds(self, query):
        return getColorProb(query)

    def compute_similarity_on_frame(self, tracks, query): # ===================== on debugging ========================== #
        loss = [0.] * len(tracks['crops'])

        with torch.no_grad():
            for t in tracks['crops']:
                l = 0.
                out = self.forward(t)
                _, pre = torch.max(out, 1) # need to calculate by sequence_len
                percentage = torch.nn.functional.softmax(out, dim=1)[0]

                color_emb = self.compute_color_embeds(query)
                for k, v in color_emb.items():
                    target = torch.LongTensor([k]*len(t)).cuda()
                    #[color_emb for i in range(size)]
                    l += self.loss_model(out, target) * v
                loss.append(l)

        return loss                                         # ===================== on debugging ========================== #

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

    def compute_color_embeds(self, queries):
        return getTypeProb(queries)

    def compute_similarity_on_frame(self, tracks, queries):
        return 1



