import collections
import itertools
import logging
import re
import types

from bert import BertModel as VidBertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers.modeling_bert import BertModel as TxtBertModel
from transformers import BertTokenizer, BertModel

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

        self.txt_tokenizer = BertTokenizer.from_pretrained(self.cfg["bert_pretrained_model"],
                                                            do_lower_case=True)
        self.txt_bert = BertModel.from_pretrained(self.cfg["bert_pretrained_model"])
        self.lang_fc = torch.nn.Linear(768, self.cfg["embedding_size"])

        self.vid_bert = VidBertModel(self.cfg["vid_bert_params"])

    def forward(self, input):
        print(type(input))
        print(len(input))

        print(input["nl"])
        print(input["frames"].shape)

        # Text Embedding

        txt_embd_list = []
        for minibatch in input["nl"]:
            token = self.txt_tokenizer(minibatch,
                                        padding=True,
                                        truncation=True,
                                        max_length=64,
                                        return_tensors="pt")
            txt_embd = self.txt_bert(**token).last_hidden_state
            txt_embd = torch.mean(txt_embd, dim=1)
            txt_embd = self.lang_fc(txt_embd)
            txt_embd_list.append(txt_embd)
        txt_embds = torch.stack(txt_embd_list).cuda()


if __name__ == '__main__':
    print(f'running script {__file__}')

    import json
    from datasets import CityFlowNLDataset
    from torch.utils.data import DataLoader, RandomSampler

    config_json = "src/config.json"

    with open(config_json) as f:
        cfg = json.load(f)

    dataset = CityFlowNLDataset(cfg["data"])
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], #cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg["train"]["num_workers"], #cfg.TRAIN.NUM_WORKERS,
                            sampler=train_sampler,
                            collate_fn=dataset.collate_fn)

    model = CEModel(cfg=cfg["model"])

    i = 0
    for batch in dataloader:
        i += 1

        model(batch)

        if i == 1:
            break