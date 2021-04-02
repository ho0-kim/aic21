import json
import time
import datetime
import csv
import logging
import argparse

import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import CityFlowNLInferenceDataset
from model import *

import numpy as np
import random
import os

def infer(args):
    # Read configuation json file
    config_json = args.config

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
    # eval_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=cfg["eval"]["batch_size"],
                            num_workers=cfg["eval"]["num_workers"],
                            collate_fn=dataset.collate_fn,
                            worker_init_fn=dataset.seed_worker)

    #model
    model_color = CarColor(cfg=cfg).cuda()
    model_type = CarType(cfg=cfg).cuda()

    ckpt = torch.load(cfg["eval"]["restore_from"],
                      map_location=lambda storage, loc: storage.cpu())
    model_color.load_state_dict(ckpt['model_color'], strict=True)
    model_type.load_state_dict(ckpt['model_type'], strict=True)
    model_color.eval()
    model_type.eval()

    time_start = time.time()
    results_dict = {}

    for idx, query_id in enumerate(queries):
        print(f'Evaluate query {idx + 1} {query_id}')
        time_eval = time.time()

        track_score = dict()
        q = queries[query_id]

        for track in dataloader:
            score_color = model_color.compute_similarity_on_frame(track, q)
            score_color = score_color.detach()

            score_type = model_type.compute_similarity_on_frame(track, q)
            score_type = score_type.detach()

            for i in range(len(track["id"])):
                track_id = track["id"][i]
                track_score[track_id] = score_color[i] + score_type[i]

        top_tracks = {k: v for k, v in sorted(track_score.items(), key=lambda item: item[1], reverse=True)}

        with open(os.path.join(cfg["eval"]["log"], "%s.log" % query_id), "w") as f:
            for k, v in top_tracks.items():
                f.write(f'{k} {v}\n')
        print(type(top_tracks.keys()))
        results_dict[query_id] = list(top_tracks.keys())
        print(f'Elapse time: {time.time() - time_eval}')
    with open(os.path.join(cfg["eval"]["log"], f"result{time.time()}.json"), "w") as f:
        json.dump(results_dict, f)
    print(f'finished.')
    print(f'Elapse time for full evaluation: {time.time() - time_start}')


if __name__ == '__main__':
    print(f'running script {__file__}')
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    infer(args=args)