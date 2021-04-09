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

from utils import motion_detection, motion_calculation
from json_formatter import NumpyArrayEncoder

def infer(args):
    # Read configuation json file
    config_json = args.config

    with open(config_json, "r") as f:
        cfg = json.load(f)

    with open(cfg["data"]["test_query_json"], "r") as f:
        queries = json.load(f)

    vicinity = Vicinity(json_path='data/test-vicinity.json', cfg=cfg)
    which_lane = RightLeftLane(json_path='data/test-2d-dir.json')
    light = TrafficLight(json_path='data/test-trafficlight_mrcnn.json')

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
                            worker_init_fn=dataset.seed_worker,
                            )

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
        colors = model_color.compute_color_list(q)
        types = model_type.compute_type_list(q)
        motion_nl = motion_detection(q) # get bit vector which explains a track behavior

        if cfg["eval"]["use_accelerator"]: # true: load files / false: save files
            # load files for accelerator start =========================
            with open(cfg["data"]["test_track_json"]) as f:
                tracks = json.load(f)

            with open(os.path.join(cfg["eval"]["metalog_color"], f"{query_id}.json"), "r") as f:
                color_probs = json.load(f)

            with open(os.path.join(cfg["eval"]["metalog_type"], f"{query_id}.json"), "r") as f:
                type_probs = json.load(f)
            # load files for accelerator end ===========================

            track_ids = list(tracks.keys())

            for track_id in track_ids:
                prct_color = np.asarray(color_probs[track_id])
                prct_type = np.asarray(type_probs[track_id])

                score_color = model_color.compute_color_prob(colors, prct_color)
                score_type = model_type.compute_type_prob(types, prct_type)
                track_score[track_id] = score_color + score_type

                if cfg["eval"]["variable_weights"]:
                    weight_overall = cfg["eval"]["overall_weight"]

                    motion_track = motion_calculation(track_id, cfg["eval"]["turn_threshold"])
                    vicinity_track, vicinity_nl = vicinity.calculation(track_id, q, model_color, model_type)
                    lane_track, lane_nl = which_lane.calculation(track_id, q)
                    light_track, light_nl = light.calculation(track_id, q)

                    score_all = motion_track + vicinity_track + lane_track + light_track
                    count_all = motion_nl + vicinity_nl + lane_nl + light_nl
                    
                    agg = np.sum(count_all)
                    if agg == 0: agg = 1e-10
                    weight_all = weight_overall * np.array(count_all) / agg # weight for right/left/spd up/spd down/stop/rear color/rear type/front color/front type/left lane/right lane

                    track_score[track_id] += np.dot(score_all, weight_all)
                else:
                    motion_track = motion_calculation(track_id, cfg["eval"]["turn_threshold"])
                    motion_score = np.dot(motion_nl, motion_track)
                    motion_weight = [.8, .8, .0, .0, .6]  # weight for right/left/spd up/spd down/stop
                    motion_score = np.dot(motion_score, motion_weight)
                    track_score[track_id] += np.sum(motion_score)

                    vicinity_score, _ = vicinity.calculation(track_id, q, model_color, model_type)
                    vicinity_weight = [.3, .3, .3, .3]  # weight for [rear color, rear type, front color, front type]
                    vicinity_score = np.dot(vicinity_score, vicinity_weight)
                    track_score[track_id] += np.sum(vicinity_score)

                    lane_score, _ = which_lane.calculation(track_id, q)
                    lane_weight = [.8, .8] # weight for [left lane, right lane]
                    lane_score = np.dot(lane_score, lane_weight)
                    track_score[track_id] += np.sum(lane_score)

                    light_score, _ = light.calculation(track_id, q)
                    light_weight = .8     # weight for traffic light
                    track_score[track_id] += light_score[0] + light_weight

        else:
            # save files for accelerator

            color_prob_dict = dict()
            type_prob_dict = dict()

            for t, track in enumerate(dataloader):
                prct_color = model_color.compute_similarity_on_frame(track)
                prct_type = model_type.compute_similarity_on_frame(track)
                track["crops"].detach()

                for i, track_id in enumerate(track["id"]):
                    score_color = model_color.compute_color_prob(colors, prct_color[i])
                    score_type = model_type.compute_type_prob(types, prct_type[i])
                    track_score[track_id] = score_color + score_type

                    if cfg["eval"]["variable_weights"]:
                        weight_overall = cfg["eval"]["overall_weight"]

                        motion_track = motion_calculation(track_id, cfg["eval"]["turn_threshold"])
                        vicinity_track, vicinity_nl = vicinity.calculation(track_id, q, model_color, model_type)
                        lane_track, lane_nl = which_lane.calculation(track_id, q)
                        light_track, light_nl = light.calculation(track_id, q)

                        score_all = motion_track + vicinity_track + lane_track + light_track
                        count_all = motion_nl + vicinity_nl + lane_nl + light_nl
                        
                        agg = np.sum(count_all)
                        if agg == 0: agg = 1e-10
                        weight_all = weight_overall * np.array(count_all) / agg # weight for right/left/spd up/spd down/stop/rear color/rear type/front color/front type/left lane/right lane

                        track_score[track_id] += np.dot(score_all, weight_all)
                    else:
                        motion_track = motion_calculation(track_id, cfg["eval"]["turn_threshold"])
                        motion_score = np.dot(motion_nl, motion_track)
                        motion_weight = [.8, .8, .0, .0, .6]  # weight for right/left/spd up/spd down/stop
                        motion_score = np.dot(motion_score, motion_weight)
                        track_score[track_id] += np.sum(motion_score)

                        vicinity_score, _ = vicinity.calculation(track_id, q, model_color, model_type)
                        vicinity_weight = [.3, .3, .3, .3]  # weight for [rear color, rear type, front color, front type]
                        vicinity_score = np.dot(vicinity_score, vicinity_weight)
                        track_score[track_id] += np.sum(vicinity_score)

                        lane_score, _ = which_lane.calculation(track_id, q)
                        lane_weight = [.8, .8] # weight for [left lane, right lane]
                        lane_score = np.dot(lane_score, lane_weight)
                        track_score[track_id] += np.sum(lane_score)

                        light_score, _ = light.calculation(track_id, q)
                        light_weight = .8     # weight for traffic light
                        track_score[track_id] += light_score[0] + light_weight

                    color_prob_dict.update({track_id: prct_color[i]})
                    type_prob_dict.update({track_id: prct_type[i]})

            # save files for eval accelerator start =====================
            with open(os.path.join(cfg["eval"]["metalog_color"], f"{query_id}.json"), "w") as f:
                json.dump(color_prob_dict, f, cls=NumpyArrayEncoder, indent=4)
            with open(os.path.join(cfg["eval"]["metalog_type"], f"{query_id}.json"), "w") as f:
                json.dump(type_prob_dict, f, cls=NumpyArrayEncoder, indent=4)
            # save files for eval accelerator end =======================

        top_tracks = {k: v for k, v in sorted(track_score.items(), key=lambda item: item[1], reverse=True)}

        with open(os.path.join(cfg["eval"]["log"], "%s.log" % query_id), "w") as f:
            for k, v in top_tracks.items():
                f.write(f'{k} {v}\n')

        results_dict[query_id] = list(top_tracks.keys())
        print(f'Elapse time: {time.time() - time_eval}')
    with open(os.path.join(cfg["eval"]["log"], f"result{time.time()}.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f'finished.')
    print(f'Elapse time for full evaluation: {time.time() - time_start}')


if __name__ == '__main__':
    print(f'running script {__file__}')
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    infer(args=args)
