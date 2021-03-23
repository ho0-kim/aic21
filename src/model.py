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
        self.cfg = cfg
        self.cfg_model = cfg["model"]
        self.cfg_data = cfg["data"]

        if self.cfg_model["use_token_type"]:
            if self.cfg_model["reducedim_method_frame"] == "simple":
                kernel_size = (self.cfg_data["frame_size"][0] - 64 + 1,
                            self.cfg_data["frame_size"][1] - 64 + 1)
                self.reducedim_fr = nn.Sequential(
                    nn.Conv2d(3, 10, kernel_size),
                    nn.Conv2d(10, 16, (33, 33)),
                    nn.Conv2d(16, 32, (17, 17)),
                    nn.Flatten(),
                    ReduceDim(32*16*16, 512)
                )
            elif self.cfg_model["reducedim_method_frame"] == "resnet":
                self.reducedim_fr = resnet18(pretrained=True)
                self.reducedim_fr.fc = nn.Sequential()

            if self.cfg_model["reducedim_method_seg"] == "simple":
                kernel_size = (self.cfg_data["frame_size"][0] - 64 + 1,
                            self.cfg_data["frame_size"][1] - 64 + 1)
                self.reducedim_se = nn.Sequential(
                    nn.Conv2d(3, 10, kernel_size),
                    nn.Conv2d(10, 16, (33, 33)),
                    nn.Conv2d(16, 32, (17, 17)),
                    nn.Flatten(),
                    ReduceDim(32*16*16, 512)
                )
            elif self.cfg_model["reducedim_method_seg"] == "resnet":
                self.reducedim_se = resnet18(pretrained=True)
                self.reducedim_se.fc = nn.Sequential()
            
            if self.cfg_model["reducedim_method_crop"] == "simple":
                kernel_size = (self.cfg_data["crop_size"][0] - 64 + 1,
                            self.cfg_data["crop_size"][1] - 64 + 1)
                self.reducedim_cr = nn.Sequential(
                    nn.Conv2d(3, 10, kernel_size),
                    nn.Conv2d(10, 16, (33, 33)),
                    nn.Conv2d(16, 32, (17, 17)),
                    nn.Flatten(),
                    ReduceDim(32*16*16, 512)
                )
            elif self.cfg_model["reducedim_method_crop"] == "resnet":
                self.reducedim_cr = resnet18(pretrained=True)
                self.reducedim_cr.fc = nn.Sequential()

            self.reducedim_hist = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 256, 512)
            )
        else:
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
                                                            do_lower_case=True,
                                                            )
        self.txt_bert = BertModel.from_pretrained(self.cfg_model["bert_pretrained_model"])
        self.lang_fc = torch.nn.Linear(768, self.cfg_model["embedding_size"])

        self.vid_bert = VidBertModel(self.cfg_model["vid_bert_params"])

    def forward(self, input):
        return {
            'vid_embds' : self.compute_vid_embds(input),
            'txt_embds' : self.compute_txt_embds(input["nl"]),
            'label' : input["label"]
        }

    def compute_vid_embds(self, input):
        batch_size = input["frames"].shape[0]
        seq_full_len = input["frames"].shape[1]

        ### feature embedding (= frames, seg, cropped, positions, etc)
        
        # 512 dimension combining all features
        features_list = []
        token_type_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        
        concat = torch.cat((input["frames"], input["segments"]), 2)
        # print(f'concatenate result {concat.shape}')

        if self.cfg_model["use_token_type"]:
            modal_feat = [[] for _ in range(len(self.cfg_data["modalities"]))]
            for i in range(batch_size):
                modal_feat[0].append(self.reducedim_fr(input["frames"][i]))
                modal_feat[1].append(self.reducedim_se(input["segments"][i]))
                modal_feat[2].append(self.reducedim_cr(input["crops"][i]))
                modal_feat[3].append(self.reducedim_hist(input["histograms"][i]))
                modal_feat[4].append(torch.hstack((input["boxes"][i], 
                                                torch.zeros((input["boxes"].shape[1], 508), 
                                                device=torch.device('cuda:0')))))
                modal_feat[5].append(torch.hstack((input["positions"][i], 
                                                torch.zeros((input["positions"].shape[1], 510), 
                                                device=torch.device('cuda:0')))))

                position_ids_list.append(torch.hstack((torch.arange(0, input["sequence_len"][i][0]), 
                                        torch.full((seq_full_len - input["sequence_len"][i][0].int(),), 0))))
                attention_mask_list.append(torch.hstack((torch.full((input["sequence_len"][i][0].int(),), 1),
                                        torch.full((seq_full_len - input["sequence_len"][i][0].int(),), 0))))

            position_ids = torch.stack(position_ids_list).to(
                                                dtype=torch.long, 
                                                device=torch.device('cuda:0'))
            attention_mask = torch.stack(attention_mask_list).cuda()

            vid_embds_list = []
            for i in range(len(self.cfg_data["modalities"])):
                _token_type_ids = torch.full((batch_size, seq_full_len,), i, device=torch.device('cuda:0'))
                features = torch.stack(modal_feat[i])
                vid_bert_output = self.vid_bert(attention_mask=attention_mask,
                                                position_ids=position_ids,
                                                token_type_ids=_token_type_ids,
                                                features=features)
                last_layer = vid_bert_output[0]
                vid_embds_list.append(last_layer[:,0])
            vid_embds = torch.stack(vid_embds_list).permute([1, 0, 2])

        else:
            for i in range(batch_size):
                features_list.append(torch.cat((self.reducedim_fs(concat[i]),
                                                self.reducedim_cr(input["crops"][i]),
                                                self.reducedim_hist(input["histograms"][i]),
                                                input["boxes"][i],
                                                input["positions"][i]), 1))
                position_ids_list.append(torch.hstack((torch.arange(0, input["sequence_len"][i][0]), 
                                        torch.full((seq_full_len - input["sequence_len"][i][0].int(),), 0))))
                attention_mask_list.append(torch.hstack((torch.full((input["sequence_len"][i][0].int(),), 1),
                                        torch.full((seq_full_len - input["sequence_len"][i][0].int(),), 0))))
                
            features = torch.stack(features_list).cuda()
            position_ids = torch.stack(position_ids_list).to(
                                                dtype=torch.long, 
                                                device=torch.device('cuda:0'))
            attention_mask = torch.stack(attention_mask_list).cuda()

            # print(f'feature embedding shape {features.shape}')
            # print(f'position_ids shape {position_ids.shape}')
            # print(f'attention mask shape {attention_mask.shape}')
            
            # Video Embedding

            vid_bert_output = self.vid_bert(attention_mask=attention_mask, 
                                            position_ids=position_ids, 
                                            features=features)
            last_layer = vid_bert_output[0]
            vid_embds = last_layer[:, 0].unsqueeze_(1)
        # print(f'video embedding shape {vid_embds.shape}')

        return vid_embds

    def compute_txt_embds(self, queries):
        # Text Embedding
        txt_embd_list = []
        for minibatch in queries:
            token = self.txt_tokenizer.batch_encode_plus(minibatch,
                                        max_length=128,
                                        padding='longest',
                                        return_tensors="pt").to(device=torch.device('cuda'))
            # txt_embd = self.txt_bert(**token).last_hidden_state
            txt_bert_output = self.txt_bert(**token)
            txt_last_layer = txt_bert_output[0]
            if self.cfg_model["text_post"] == 'mean':
                txt_embd = torch.mean(txt_last_layer, dim=1)
            elif self.cfg_model["text_post"] == 'max':
                txt_embd, _ = torch.max(txt_last_layer, dim=1)
            txt_embd = self.lang_fc(txt_embd)
            txt_embd_list.append(txt_embd)
        txt_embds = torch.stack(txt_embd_list).cuda()
        # print(f'text embedding shape {txt_embds.shape}')

        return txt_embds

    def compute_similarity(self, vid_embds, txt_embds):
        if self.cfg["loss"]["type"] == "BinaryCrossEntropy":
            if self.cfg_model["use_token_type"]:
                pass
            else:
                sims = []
                for i in range(len(txt_embds[0])):
                    d = F.pairwise_distance(vid_embds, txt_embds[:, i, :])
                    sims.append(torch.mean(torch.exp(-d), dim=1))
                sims = torch.sum(torch.stack(sims).permute([1, 0]), dim=1, keepdim=True)
        else:   # For MaxMarginRankingLoss
            batch_size = vid_embds.size()[0]
            if self.cfg_model["use_token_type"]:
                pass
            else:
                sims = []
                for i in range(batch_size):
                    sim = torch.matmul(vid_embds[i], txt_embds.permute([0, 2, 1]))
                    sim = torch.mean(sim, dim=2)
                    sims.append(sim)
                sims = torch.stack(sims).squeeze()
        return sims

    def compute_similarity_for_eval(self, tracks, queries):
        vid_embds = self.compute_vid_embds(tracks)
        txt_embds = self.compute_txt_embds(queries)
        if self.cfg["loss"]["type"] == "BinaryCrossEntropy":
            return self.compute_similarity(vid_embds, txt_embds).squeeze()
        else:   # For MaxMarginRankingLoss
            s = self.compute_similarity(vid_embds, txt_embds)
            if tracks["frames"].size()[0] == 1:
                return s
            else:
                return torch.diag(s)

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


if __name__ == '__main__':
    print(f'running script {__file__}')

    import json
    from datasets import CityFlowNLDataset
    from torch.utils.data import DataLoader, RandomSampler

    from loss import get_loss_model

    # config_json = "src/config.json"
    config_json = "src/maxmargin.json"

    with open(config_json) as f:
        cfg = json.load(f)

    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.multiprocessing.set_start_method('spawn')

    dataset = CityFlowNLDataset(cfg)
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], #cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg["train"]["num_workers"], #cfg.TRAIN.NUM_WORKERS,
                            sampler=train_sampler,
                            collate_fn=dataset.collate_fn,
                            worker_init_fn=dataset.seed_worker)

    model = CEModel(cfg=cfg).cuda()#.to(torch.device('cuda:0'))
    loss_model = get_loss_model(cfg)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_optimizer(cfg, trainable_params)

    epochs = 10
    for e in range(epochs):
        b = 0
        for batch in dataloader:
            b += 1
            
            optimizer.zero_grad()

            out = model(batch)

            similarity = model.compute_similarity(out["vid_embds"], out["txt_embds"])
            loss = loss_model(similarity, out["label"])
            print(f'Eposch : {e}  Batch : {b} Loss : {loss}')

            sim = model.compute_similarity_for_eval(batch)
            print(f'similarity for eval: {sim}')

            loss.backward()
            optimizer.step()

            # del loss
            # del similarity
            # del out

            # torch.cuda.empty_cache()