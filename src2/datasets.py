import json
import os
import random
from collections import Counter

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, dataloader
from utils import *

class CityFlowNLDataset(Dataset):
    def __init__(self, cfg):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.cfg = cfg
        self.seed = cfg["seed"]
        self.data_cfg = cfg["data"]
        with open(self.data_cfg["train_json"]) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        for track in self.list_of_tracks:
            for frame_idx, frame in enumerate(track["frames"]):
                track["frames"][frame_idx] = os.path.join(self.data_cfg["cityflow_path"], frame)

    def __len__(self):
        return len(self.list_of_uuids)

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        track = self.list_of_tracks[index]

        seq_len = len(track["frames"])
        frame_index = int(random.uniform(0, seq_len-1))

        frame = cv2.imread(track["frames"][frame_index])
        box = track["boxes"][frame_index]
        crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
        crop = cv2.resize(crop, dsize=tuple(self.data_cfg["crop_size"]))  # d: 128, 128, 3

        colors = list()
        types = list()
        color_count = 0
        type_count = 0
        for s in track["nl"]:
            tokens = tokenize(s)
            color, color_label = getColorLabel(tokens)
            type, type_label = getTypeLabel(tokens)
            colors.append(color_label)
            types.append(type_label)

            if color_label >= 0:
                color_count += 1
            if type_label >= 0:
                type_count += 1

        assert(color_count >= 0)
        assert(type_count >= 0)

        color_prob = dict()
        type_prob = dict()
        color_counter = Counter(colors)
        type_counter = Counter(types)

        for color, count in color_counter.items():
            if color >= 0:
                color_prob.update({color: count / color_count})

        for type, count in type_counter.items():
            if type >= 0:
                type_prob.update({type: count / type_count})

        crop = torch.from_numpy(crop).permute([2, 0, 1]).cuda()

        dp = {}
        dp["crop"] = crop
        dp["color"] = color_prob
        dp["type"] = type_prob

        return dp

    def collate_fn(self, batch):
        ret = {}
        ret["crop"] = torch.stack([b["crop"] for b in batch]).to(dtype=torch.float32).cuda()
        ret["color"] = [b["color"] for b in batch]
        ret["type"] = [b["type"] for b in batch]
        return ret

    def seed_worker(self, worker_id):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)