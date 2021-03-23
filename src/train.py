import json
import time
import datetime
import csv
import logging
import argparse

import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import CityFlowNLDataset
from loss import get_loss_model
from model import CEModel, get_optimizer

import numpy as np
import random

def float2timeformat(seconds):
    s = seconds % 60
    m = seconds % 3600 // 60
    h = seconds // 3600
    return "%2d:%2d:%2f" % (h, m, s)

def train(args):
    # Set logger
    log_file = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=f'{log_file}.log',
                        filemode='w')
    logger = logging.getLogger(__name__)

    # Set CSV log file
    csv_file = open(f'{log_file}.csv', 'w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['epoch', 'batch', 'loss'])


    # Read configuation json file
    config_json = args.config

    with open(config_json) as f:
        cfg = json.load(f)

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
    dataset = CityFlowNLDataset(cfg)
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"],
                            num_workers=cfg["train"]["num_workers"],
                            sampler=train_sampler,
                            collate_fn=dataset.collate_fn,
                            worker_init_fn=dataset.seed_worker)

    # load model, loss, optimizer
    model = CEModel(cfg=cfg).cuda()
    loss_model = get_loss_model(cfg)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_optimizer(cfg, trainable_params)

    logger.debug(f'configuration file : {config_json}')

    epoch_start = 0
    if args.pretrained != None:
        ckpt = torch.load(args.pretrained)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch_start = ckpt['epoch']

    data_size = len(dataset)
    num_batch = int(data_size / cfg["train"]["batch_size"])

    time_start = time.time()
    logger.debug(f'trainig start')
    for e in range(epoch_start, cfg["train"]["num_epochs"]):
        model.train()
        b = 0
        time_e_start = time.time()
        losses = 0
        for batch in dataloader:
            time_b_start = time.time()
            b += 1
            
            optimizer.zero_grad()

            out = model(batch)

            similarity = model.compute_similarity(out["vid_embds"], out["txt_embds"])
            loss = loss_model(similarity, out["label"])
            losses += loss

            loss.backward()
            optimizer.step()

            time_current = time.time()
            elapse_b = time_current - time_b_start
            elapse_e = time_current - time_e_start
            elapse_f = time_current - time_start
            msg = f'Epoch : {e}  Batch : {b}/{num_batch} Loss : {loss}' + \
                    f' Elapse time : (batch) {float2timeformat(elapse_b)}' + \
                                f' (epoch) {float2timeformat(elapse_e)}' + \
                                f' (full) {float2timeformat(elapse_f)}'
            logger.debug(msg)
            print(msg)
            csv_writer.writerow([e+1, b, loss.cpu().detach().numpy()])
        msg = f'Epoch : {e}  Average Loss : {losses / num_batch}' + \
                    f' Elapse time : (epoch) {float2timeformat(elapse_e)}' + \
                                f' (full) {float2timeformat(elapse_f)}'
        print(msg)
        logger.debug(msg)

        # Save model
        if e % 5 == 0:
            torch.save({
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, log_file+f'_epoch{e}.pt')

if __name__ == '__main__':
    print(f'running script {__file__}')

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-p', '--pretrained', default=None)
    args = parser.parse_args()

    train(args=args)