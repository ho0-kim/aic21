import collections
import itertools
import logging
import re
import types

from bert import BertModel as VidBertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
# from transformers.modeling_bert import BertModel as TxtBertModel
from transformers import BertTokenizer, BertModel

import numpy as np
import random

class ReduceDim(nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x) 
        x = F.normalize(x, dim=-1)
        return x

class CEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg_model = cfg["model"]
        self.cfg_data = cfg["data"]

        if self.cfg_model["reducedim_method_frame"] == "simple":
            kernel_size = (self.cfg_data["frame_size"][0] - 64 + 1,
                        self.cfg_data["frame_size"][1] - 64 + 1)
            self.reducedim_fs = nn.Sequential(
                nn.Conv2d(6, 10, kernel_size),
                nn.Conv2d(10, 16, (33, 33)),
                nn.Conv2d(16, 32, (17, 17)),
                nn.Flatten(),
                ReduceDim(32*16*16, 256)
            )
        elif self.cfg_model["reducedim_method_frame"] == "resnet":
            kernel_size = (self.cfg_data["frame_size"][0] - 224 + 1,
                        self.cfg_data["frame_size"][1] - 224 + 1)
            resnet_fs = resnet18(pretrained=True)
            resnet_fs.fc = nn.Linear(512, 256) #nn.Sequential()
            self.reducedim_fs = nn.Sequential(
                nn.Conv2d(6, 3, kernel_size),
                resnet_fs
            )

        if self.cfg_model["reducedim_method_crop"] == "simple":
            kernel_size = (self.cfg_data["crop_size"][0] - 64 + 1,
                        self.cfg_data["crop_size"][1] - 64 + 1)
            self.reducedim_cr = nn.Sequential(
                nn.Conv2d(3, 10, kernel_size),
                nn.Conv2d(10, 16, (33, 33)),
                nn.Conv2d(16, 32, (17, 17)),
                nn.Flatten(),
                ReduceDim(32*16*16, 128)
            )
        elif self.cfg_model["reducedim_method_crop"] == "resnet":
            kernel_size = (self.cfg_data["crop_size"][0] - 64 + 1,
                        self.cfg_data["crop_size"][1] - 64 + 1)
            resnet_cr = resnet18(pretrained=True)
            resnet_cr.fc = nn.Linear(512, 128)
            self.reducedim_cr = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size),
                resnet_cr
            )

        self.reducedim_hist = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 256, 122)
        )        


        self.txt_tokenizer = BertTokenizer.from_pretrained(self.cfg_model["bert_pretrained_model"],
                                                            do_lower_case=True)
        self.txt_bert = BertModel.from_pretrained(self.cfg_model["bert_pretrained_model"])
        self.lang_fc = torch.nn.Linear(768, self.cfg_model["embedding_size"])

        self.vid_bert = VidBertModel(self.cfg_model["vid_bert_params"])

    def forward(self, input):
        print(f'input type {type(input)}')
        print(f'input[frames] shape {input["frames"].shape}')
        print(f'input[frames] type {type(input["frames"])}')
        print(f'input[segments] shape {input["segments"].shape}')
        print(f'input[crops] shape {input["crops"].shape}')
        print(f'input[histograms] shape {input["histograms"].shape}')
        print(f'input[boxes] shape {input["boxes"].shape}')
        print(f'input[positions] shape {input["positions"].shape}')
        print(f'input[sequence_len] shape {input["sequence_len"].shape}')

        ### feature embedding (= frames, seg, cropped, positions, etc)
        
        # 512 dimension combining all features
        features_list = []
        attention_mask_list = []
        position_ids_list = []
        
        concat = torch.cat((input["frames"], input["segments"]), 2)
        print(f'concatenate result {concat.shape}')
        for i in range(input["frames"].shape[0]):
            features_list.append(torch.cat((self.reducedim_fs(concat[i]),
                                            self.reducedim_cr(input["crops"][i]),
                                            self.reducedim_hist(input["histograms"][i]),
                                            input["boxes"][i],
                                            input["positions"][i]), 1))
            position_ids_list.append(torch.hstack((torch.arange(0, input["sequence_len"][i][0]), 
                                    torch.full((input["frames"].shape[1] - input["sequence_len"][i][0].int(),), 0))))
            attention_mask_list.append(torch.hstack((torch.full((input["sequence_len"][i][0].int(),), 1),
                                    torch.full((input["frames"].shape[1] - input["sequence_len"][i][0].int(),), 0))))
            
        features = torch.stack(features_list).cuda()
        position_ids = torch.stack(position_ids_list).to(
                                            dtype=torch.long, 
                                            device=torch.device('cuda:0'))
        attention_mask = torch.stack(attention_mask_list).cuda()

        print(f'feature embedding shape {features.shape}')
        print(f'position_ids shape {position_ids.shape}')
        print(f'attention mask shape {attention_mask.shape}')
        
        # Video Embedding

        vid_bert_output = self.vid_bert(attention_mask, position_ids, features)
        last_layer = vid_bert_output[0]
        vid_embd = last_layer[:, 0]
        print(f'video embedding shape {vid_embd.shape}')

        # Text Embedding

        txt_embd_list = []
        for minibatch in input["nl"]:
            token = self.txt_tokenizer(minibatch,
                                        padding=True,
                                        truncation=True,
                                        max_length=64,
                                        return_tensors="pt").to(device=torch.device('cuda'))
            txt_embd = self.txt_bert(**token).last_hidden_state
            txt_embd = torch.mean(txt_embd, dim=1)
            txt_embd = self.lang_fc(txt_embd)
            txt_embd_list.append(txt_embd)
        txt_embds = torch.stack(txt_embd_list).cuda()
        print(f'text embedding shape {txt_embds.shape}')

        return {
            'vid_embd' : vid_embd,
            'txt_embds' : txt_embds,
        }


if __name__ == '__main__':
    print(f'running script {__file__}')

    import json
    from datasets import CityFlowNLDataset
    from torch.utils.data import DataLoader, RandomSampler

    config_json = "src/config.json"

    with open(config_json) as f:
        cfg = json.load(f)

    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    torch.multiprocessing.set_start_method('spawn')

    dataset = CityFlowNLDataset(cfg)
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], #cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg["train"]["num_workers"], #cfg.TRAIN.NUM_WORKERS,
                            sampler=train_sampler,
                            collate_fn=dataset.collate_fn,
                            worker_init_fn=dataset.seed_worker)

    model = CEModel(cfg=cfg).cuda()#.to(torch.device('cuda:0'))

    i = 0
    for batch in dataloader:
        i += 1

        out = model(batch)

        print(out)

        if i == 1:
            break