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

        colors = list()                         # -------------- getColorProb(nls)
        types = list()                          # -------------- getTypeProb(nls)
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
                type_prob.update({type: count / type_count})    #================================

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

class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, cfg):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.seed = cfg["seed"]
        self.data_cfg = cfg["data"]
        with open(self.data_cfg["test_track_json"]) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        # self._logger = get_logger()
        for track in self.list_of_tracks:
            for frame_idx, frame in enumerate(track["frames"]):
                track["frames"][frame_idx] = os.path.join(self.data_cfg["cityflow_path"], frame)

    def __len__(self):
        return len(self.list_of_uuids)

    def __getitem__(self, index):
        """
        :return: a dictionary for each track:
        id, frames, boxes: uuid, frames, boxes for the track
        """
        track = self.list_of_tracks[index]
        seq_len = len(track["frames"])

        if seq_len > self.data_cfg["max_seq_len"]:
            seq_len = self.data_cfg["max_seq_len"]

        test_seq = random.sample(range(len(track["frames"])), seq_len) # random sampling without duplicate

        frames = list()
        crops = list()

        for i in test_seq:
            frame_idx = i
            frame_path = track["frames"][frame_idx]

            frame = cv2.imread(frame_path)
            box = track["boxes"][frame_idx]
            crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]

            frame = cv2.resize(frame, dsize=tuple(self.data_cfg["frame_size"]))
            crop = cv2.resize(crop, dsize=tuple(self.data_cfg["crop_size"]))  # d: 128, 128, 3

            frames.append(torch.from_numpy(frame).permute([2, 0, 1]).cuda())
            crops.append(torch.from_numpy(crop).permute([2, 0, 1]).cuda())

        dp = {"id": self.list_of_uuids[index]}
        dp.update({"frames": frames})
        dp.update({"crops": crops})

        return dp

    def collate_fn(self, batch):
        # padding for batch
        lengths = [len(t["frames"]) for t in batch]
        max_len = max(lengths)

        for index_in_batch, data in enumerate(batch):
            for _ in range(max_len - lengths[index_in_batch]):
                #data["frames"].append(torch.zeros_like(data["frames"][0]).cuda())
                data["crops"].append(torch.zeros_like(data["crops"][0]).cuda())

            data["sequence_len"] = torch.Tensor([lengths[index_in_batch]]).cuda()
            #data["frames"] = torch.stack(data["frames"]).cuda()
            data["crops"] = torch.stack(data["crops"]).cuda()

        ret = {}
        ret["id"] = [b["id"] for b in batch]
        ret["sequence_len"] = torch.stack([data["sequence_len"] for data in batch]).to(dtype=torch.float32).cuda()
        #ret["frames"] = torch.stack([data["frames"] for data in batch]).to(dtype=torch.float32).cuda()
        ret["crops"] = torch.stack([data["crops"] for data in batch]).to(dtype=torch.float32).cuda()

        return ret

    def seed_worker(self, worker_id):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)