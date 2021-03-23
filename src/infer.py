import json
import time
import datetime
import csv
import logging

import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import CityFlowNLInferenceDataset
from loss import get_loss_model
from model import CEModel, get_optimizer

import numpy as np
import random
import os


def float2timeformat(seconds):
    s = seconds % 60
    m = seconds % 3600 // 60
    h = seconds // 3600
    return "%2d:%2d:%2f" % (h, m, s)


def infer():
    # Read configuation json file
    config_json = "src/config.json"

    with open(config_json, "r") as f:
        cfg = json.load(f)

    with open(cfg["data"]["test_query_json"], "r") as f:
        queries = json.load(f)

    # save and load files(??)
    # if os.path.isdir(cfg["eval"]["continue"]):
    #     files = os.listdir(os.path.join(cfg["eval"]["continue"], "logs"))
    #     for q in files:
    #         del queries[q.split(".")[0]]
    #     cfg

    # Set random seed
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # cudnn set
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # multi-processing set
    torch.multiprocessing.set_start_method('spawn')

    # load data
    dataset = CityFlowNLInferenceDataset(cfg)
    ckpt = torch.load(cfg["eval"]["restore_from"],
                      map_location=lambda storage, loc: storage.cpu())
    model = CEModel(cfg=cfg)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    model = model.cuda()

    #eval_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=cfg["eval"]["batch_size"],
                            num_workers=cfg["eval"]["num_workers"],
                            collate_fn=dataset.collate_fn,
                            worker_init_fn=dataset.seed_worker)

    for idx, query_id in enumerate(queries):
        print(f'Evaluate query {query_id}')
        track_score = dict()
        q = queries[query_id]
        q = [q for i in range(cfg["eval"]["batch_size"])]

        count = 0       # (temporary code)

        for track in dataloader:
            s = model.compute_similarity_for_eval(track, q)
            s = s.detach()
            track_id = track["id"][0]
            track_score[track_id] = s.item()

            if count % 10 == 9:         # (temporary code)
                break                   # (temporary code)
            count += 1                  # (temporary code)

        top_tracks = sorted(track_score, key=track_score.get, reverse=True)
        with open(os.path.join(cfg["eval"]["log"], "%s.log" % query_id), "w") as f:
            for track in top_tracks:
                f.write(f'{track}\n')
    print(f'finished.')

if __name__ == '__main__':
    print(f'running script {__file__}')
    infer()
