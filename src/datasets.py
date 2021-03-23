import json
import os
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence


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

    def __len__(self):
        return len(self.list_of_uuids)

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        if random.uniform(0, 1) > self.data_cfg["positive_threshold"]:
            label = 1
        else:
            label = 0

        if self.cfg["loss"]["type"] == "MaxMarginRankingLoss":
            label = 1

        track = self.list_of_tracks[index]
        if label == 0:
            nl_index = random.choice([ x for x in range(len(self.list_of_tracks)) if x != index])
        else:
            nl_index = index
        nl_track = self.list_of_tracks[nl_index]

        frames = []
        crops = []
        histograms = []
        segments = []
        boxes = []
        positions = []

        seq_len = len(track["frames"])
        if seq_len > self.data_cfg["max_seq_len"]:
            skip = seq_len // (self.data_cfg["max_seq_len"] / 2)
        else:
            skip = 1

        for i, frame_path in enumerate(track["frames"]):
            if i % skip == 0: # Skip frames
            
                frame = cv2.imread(frame_path)
                frm_h, frm_w, _ = frame.shape
                box = track["boxes"][i]
                crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
                crop = cv2.resize(crop, dsize=tuple(self.data_cfg["crop_size"]))
                hist = [cv2.calcHist([crop],[0],None,[256],[0,256]), # B
                        cv2.calcHist([crop],[1],None,[256],[0,256]), # G
                        cv2.calcHist([crop],[2],None,[256],[0,256])] # R

                img_file = os.path.basename(frame_path)
                img_name = img_file.split(".")[0]
                seg_path = os.path.dirname(os.path.dirname(frame_path))
                seg_path = os.path.join(seg_path, "seg")
                seg_path = os.path.join(seg_path, img_name+"_prediction.png")
                segmented = cv2.imread(seg_path)

                frame = cv2.resize(frame, dsize=tuple(self.data_cfg["frame_size"]))
                segmented = cv2.resize(segmented, dsize=tuple(self.data_cfg["frame_size"]))

                box[0] = box[0] / frm_w
                box[2] = box[2] / frm_w
                box[1] = box[1] / frm_h
                box[3] = box[3] / frm_h
                
                frames.append(torch.from_numpy(frame).permute([2, 0, 1]).cuda())
                crops.append(torch.from_numpy(crop).permute([2, 0, 1]).cuda())
                histograms.append(torch.FloatTensor(hist).cuda())
                positions.append(torch.FloatTensor([box[0], box[1]]).cuda())
                segments.append(torch.from_numpy(segmented).permute([2, 0, 1]).cuda())
                boxes.append(torch.FloatTensor(box).cuda())

        dp = {}
        dp["frames"] = frames
        dp["crops"] = crops
        dp["histograms" ] = histograms
        dp["segments"] = segments
        dp["boxes"] = boxes
        dp["positions"] = positions
        dp["label"] = torch.Tensor([label]).to(dtype=torch.float32).cuda()
        dp["nl"] = random.sample(nl_track["nl"], 3)

        return dp

    def collate_fn(self, batch):
        # Zero pads
        lengths = [ len(t["frames"]) for t in batch ]
        max_len = max(lengths)

        for i, b in enumerate(batch):
            for _ in range(max_len - lengths[i]):
                b["frames"].append(torch.zeros_like(b["frames"][0]).cuda())
                b["crops"].append(torch.zeros_like(b["crops"][0]).cuda())
                b["histograms"].append(torch.zeros_like(b["histograms"][0]).cuda())
                b["segments"].append(torch.zeros_like(b["segments"][0]).cuda())
                b["boxes"].append(torch.zeros_like(b["boxes"][0]).cuda())
                b["positions"].append(torch.zeros_like(b["positions"][0]).cuda())

            b["sequence_len"] = torch.Tensor([lengths[i]]).cuda()

            b["frames"] = torch.stack(b["frames"]).cuda()
            b["crops"] = torch.stack(b["crops"]).cuda()
            b["histograms"] = torch.stack(b["histograms"]).cuda()
            b["segments"] = torch.stack(b["segments"]).cuda()
            b["boxes"] = torch.stack(b["boxes"]).cuda()
            b["positions"] = torch.stack(b["positions"]).cuda()

        ret = {}
        ret["frames"] = torch.stack([ b["frames"] for b in batch ]).to(
                dtype=torch.float32).cuda()
        ret["crops"] = torch.stack([ b["crops"] for b in batch ]).to(
                dtype=torch.float32).cuda()
        ret["histograms"] = torch.stack([ b["histograms"] for b in batch ]).to(
                dtype=torch.float32).cuda()
        ret["segments"] = torch.stack([ b["segments"] for b in batch ]).to(
                dtype=torch.float32).cuda()
        ret["boxes"] = torch.stack([ b["boxes"] for b in batch ]).to(
                dtype=torch.float32).cuda()
        ret["positions"] = torch.stack([ b["positions"] for b in batch ]).to(
                dtype=torch.float32).cuda()
        ret["label"] = torch.stack([ b["label"] for b in batch ]).to(
                dtype=torch.float32).cuda()
        ret["nl"] = [ b["nl"] for b in batch ]
        ret["sequence_len"] = torch.stack([ b["sequence_len"] for b in batch ]).to(
                dtype=torch.float32).cuda()

        ret["frames"] = ret["frames"] / 255.
        ret["crops"] = ret["crops"] / 255.
        ret["segments"] = ret["segments"] / 255.

        ret["histograms"] = ret["histograms"] / 2025.

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

    def __len__(self):
        return len(self.list_of_uuids)

    def __getitem__(self, index):
        """
        :return: a dictionary for each track:
        id, frames, boxes: uuid, frames, boxes for the track
        """
        frames = []
        boxes = []
        crops = []
        histograms = []
        segments = []
        positions = []

        track = self.list_of_tracks[index]
        seq_len = len(track["frames"])
        if seq_len > self.data_cfg["max_seq_len"]:
            skip = seq_len // (self.data_cfg["max_seq_len"] / 2) # might be removed.
        else:
            skip = 1

        for frame_idx, frame_path in enumerate(track["frames"]):
            if frame_idx % skip == 0:   # skip frames
                frame = cv2.imread(frame_path)
                frm_h, frm_w, _ = frame.shape
                box = track["boxes"][frame_idx]
                crop = frame[box[1]: box[1]+box[3], box[0]: box[0]+box[2], :]
                crop = cv2.resize(crop, dsize=tuple(self.data_cfg["crop_size"]))
                hist = [cv2.calcHist([crop], [0], None, [256], [0, 256]),   # B  # <----before permute
                        cv2.calcHist([crop], [1], None, [256], [0, 256]),   # G
                        cv2.calcHist([crop], [2], None, [256], [0, 256])]   # R

                img_file = os.path.basename(frame_path)
                img_name = img_file.split(".")[0]
                seg_path = os.path.dirname(os.path.dirname(frame_path))
                seg_path = os.path.join(seg_path, "seg")
                seg_path = os.path.join(seg_path, img_name+"_prediction.png")
                segmented = cv2.imread(seg_path)

                frame = cv2.resize(frame, dsize=tuple(self.data_cfg["frame_size"]))
                segmented = cv2.resize(segmented, dsize = tuple(self.data_cfg["frame_size"]))

                box[0] = box[0] / frm_w     # 1, 3: related to height // 0, 2: related to width
                box[2] = box[2] / frm_w
                box[1] = box[1] / frm_h
                box[3] = box[3] / frm_h

                frames.append(torch.from_numpy(frame).permute([2, 0, 1]).cuda())
                boxes.append(torch.FloatTensor(box).cuda())
                crops.append(torch.from_numpy(crop).permute([2, 0, 1]).cuda())
                histograms.append(torch.FloatTensor(hist).cuda())
                segments.append(torch.from_numpy(segmented).permute([2, 0, 1]).cuda())
                positions.append(torch.FloatTensor([box[0], box[1]]).cuda()) # <----y, x :why not centroid??

        dp = {"id": self.list_of_uuids[index]}
        dp.update({"frames": frames})
        dp.update({"boxes": boxes})
        dp.update({"crops": crops})             # <---- different from baseline; stacking in collate_fn
        dp.update({"histograms": histograms})
        dp.update({"segments": segments})
        dp.update({"positions": positions})

        return dp

    def collate_fn(self, batch):
        # padding for batch
        lengths = [len(t["frames"]) for t in batch]
        max_len = max(lengths)

        for index_in_batch, data in enumerate(batch):
            for _ in range(max_len - lengths[index_in_batch]):
                data["frames"].append(torch.zeros_like(data["frames"][0]).cuda())
                data["boxes"].append(torch.zeros_like(data["boxes"][0]).cuda())
                data["crops"].append(torch.zeros_like(data["crops"][0]).cuda())
                data["histograms"].append(torch.zeros_like(data["histograms"][0]).cuda())
                data["segments"].append(torch.zeros_like(data["segments"][0]).cuda())
                data["positions"].append(torch.zeros_like(data["positions"][0]).cuda())

            data["sequence_len"] = torch.Tensor([lengths[index_in_batch]]).cuda()
            data["frames"] = torch.stack(data["frames"]).cuda()
            data["boxes"] = torch.stack(data["boxes"]).cuda()
            data["crops"] = torch.stack(data["crops"]).cuda()
            data["histograms"] = torch.stack(data["histograms"]).cuda()
            data["segments"] = torch.stack(data["segments"]).cuda()
            data["positions"] = torch.stack(data["positions"]).cuda()

        ret = {}
        ret["id"] = [b["id"] for b in batch]
        ret["sequence_len"] = torch.stack([data["sequence_len"] for data in batch]).to(dtype=torch.float32).cuda()
        ret["frames"] = torch.stack([data["frames"] for data in batch]).to(dtype=torch.float32).cuda()
        ret["boxes"] = torch.stack([data["boxes"] for data in batch]).to(dtype=torch.float32).cuda()
        ret["crops"] = torch.stack([data["crops"] for data in batch]).to(dtype=torch.float32).cuda()
        ret["histograms"] = torch.stack([data["histograms"] for data in batch]).to(dtype=torch.float32).cuda()
        ret["segments"] = torch.stack([data["segments"] for data in batch]).to(dtype=torch.float32).cuda()
        ret["positions"] = torch.stack([data["positions"] for data in batch]).to(dtype=torch.float32).cuda()

        ret["frames"] = ret["frames"] / 255.
        ret["crops"] = ret["crops"] / 255.
        ret["segments"] = ret["segments"] / 255.

        ret["histograms"] = ret["histograms"] / 2025. # <---- why 2025???

        return ret

    def seed_worker(self, worker_id):
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

# def collate_fn(batch):
#     # if isinstance(batch, BBoxList):
#         # return batch
#     return batch
#     # return default_collate(batch)

if __name__ == '__main__':
    config_json = "src/config.json"

    with open(config_json) as f:
        conf = json.load(f)

    # print(conf)
    # print(conf["data"])
    # print(conf["data"]["train_json"])

    # data_conf = conf["data"]
    # print(data_conf["train_json"])


    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
    from torch.nn.parallel import DistributedDataParallel

    # rank = 0
    # dist_rank = 0
    # WORLD_SIZE = 1
    # INIT_METHOD = "file://../baseline/shared_file.txt" #"tcp://%s:%d" % ("127.0.0.1", 12000)
    # dist.init_process_group(backend="gloo", rank=dist_rank,
    #                         world_size=WORLD_SIZE,
    #                         init_method=INIT_METHOD)

    cfg = conf

    dataset = CityFlowNLDataset(cfg)

    # train_sampler = DistributedSampler(dataset)
    # train_sampler = SequentialSampler(dataset)
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], #cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg["train"]["num_workers"], #cfg.TRAIN.NUM_WORKERS,
                            sampler=train_sampler,
                            collate_fn=dataset.collate_fn,
                            worker_init_fn=dataset.seed_worker)
    print('dataloader set')

    i = 0
    for minibatch in dataloader:
        i += 1
        
        print(type(minibatch))
        print(len(minibatch))
        tracks = minibatch
        print(type(tracks[0]["frames"]))
        print(tracks[0]["frames"][-1].shape)

        # print(track["frames"][0].shape)
        # print(track["crops"][0].shape)
        # print(track["histograms"][0].shape)
        # print(track["segments"][0].shape)
        # print(track["boxes"][0].shape)
        # print(track["positions"][0].shape)
        # print(track["label"][0].shape)

        # print(len(track["frames"]))
        # print(len(track["crops"]))
        # print(len(track["histograms"]))
        # print(len(track["segments"]))
        # print(len(track["boxes"]))
        # print(len(track["positions"]))
        # print(len(track["label"]))

        # print(track["nl"])

        print(f'iter: {i}')

        if i == 1:
            break
        # dp["frames"] = frames
        # dp["crops"] = crops
        # dp["histograms" ] = histograms
        # dp["segments"] = segments
        # dp["boxes"] = track["boxes"]
        # dp["positions"] = positions
        # dp["label"] = torch.Tensor([label]).to(dtype=torch.float32)

        #OSError: [Errno 24] Too many open files
        #solution: ulimit -n 10240
