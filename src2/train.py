import os
import argparse
import logging
import time
import datetime
import csv
import json
import numpy as np
import random
import math

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, models, transforms
from datasets import CityFlowNLDataset
from model import *

# need normalization??

'''
data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
def float2timeformat(seconds):
    s = seconds % 60
    m = seconds % 3600 // 60
    h = seconds // 3600
    return "%2d:%2d:%2f" % (h, m, s)

def train(args):
    os.makedirs('logs', exist_ok=True)
    os.makedirs('ckpts', exist_ok=True)

    # Set logger
    log_file = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=f'logs/{log_file}.log',
                        filemode='w')
    logger = logging.getLogger(__name__)

    # Set CSV log file
    csv_file = open(f'logs/{log_file}.csv', 'w')
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
    model_color = CarColor(cfg=cfg).cuda()
    model_type = CarType(cfg=cfg).cuda()

    # loss function
    loss_model = get_loss_model(cfg) #criterion
    # check if all parameters are optimized
    optimizer_color = get_optimizer(cfg, model_color.parameters())
    optimizer_type = get_optimizer(cfg, model_type.parameters())
    # decrease learning rate per defined epoch
    exp_lr_scheduler_color =get_lr_scheduler(cfg, optimizer_color)
    exp_lr_scheduler_type = get_lr_scheduler(cfg, optimizer_type)

    logger.debug(f'configuration file : {config_json}')

    epoch_start = 0
    # ----------------------------------??
    if args.pretrained != None:
        ckpt = torch.load(args.pretrained)
        model_color.load_state_dict(ckpt['model_color'])
        optimizer_color.load_state_dict(ckpt['optimizer_color']) # optimizer zero_grad() ???????
        model_type.load_state_dict(ckpt['model_type'])
        optimizer_type.load_state_dict(ckpt['optimizer_type'])  # optimizer zero_grad() ???????
        epoch_start = ckpt['epoch']
    # ----------------------------------??

    data_size = len(dataset)
    num_batch = int(data_size / cfg["train"]["batch_size"])

    time_start = time.time()
    logger.debug(f'trainig start')
    model_color.train() # <---- training mode
    model_type.train()  # <---- training mode

    for e in range(epoch_start, cfg["train"]["num_epochs"]):
        b = 0
        time_e_start = time.time()
        losses_color = 0
        losses_type = 0
        for batch in dataloader:
            time_b_start = time.time()
            b += 1

            # 1. train car color model
            optimizer_color.zero_grad()
            loss_color = model_color.compute_loss(batch)

            if not (math.isnan(loss_color.data.item())
                    or math.isinf(loss_color.data.item())
                    or loss_color.data.item() > cfg["train"]["loss_clip_value"]):
                        loss_color.backward()
                        optimizer_color.step()
                        losses_color += loss_color.item() * batch['crop'].size(0)

            # 2. train car type model
            optimizer_type.zero_grad()
            loss_type = model_type.compute_loss(batch)

            if not (math.isnan(loss_type.data.item())
                    or math.isinf(loss_type.data.item())
                    or loss_type.data.item() > cfg["train"]["loss_clip_value"]):
                        loss_type.backward()
                        optimizer_type.step()
                        losses_type += loss_type.item() * batch['crop'].size(0)

            time_current = time.time()
            elapse_b = time_current - time_b_start
            elapse_e = time_current - time_e_start
            elapse_f = time_current - time_start
            msg = f'Epoch : {e}  Batch : {b}/{num_batch}' + \
                  f' Loss_Color : {loss_color} Loss_Type : {loss_type}' + \
                  f' Elapse time : (batch) {float2timeformat(elapse_b)}' + \
                  f' (epoch) {float2timeformat(elapse_e)}' + \
                  f' (full) {float2timeformat(elapse_f)}'
            logger.debug(msg)
            print(msg)
            csv_writer.writerow([e + 1, b, loss_color.cpu().detach().numpy(), loss_type.cpu().detach().numpy()])

        exp_lr_scheduler_color.step()
        exp_lr_scheduler_type.step()
        msg = f'Epoch : {e}  Average Loss for car color: {losses_color / data_size}' + \
              F'Average Loss for car type: {losses_type / data_size}' + \
              f' Elapse time : (epoch) {float2timeformat(elapse_e)}' + \
              f' (full) {float2timeformat(elapse_f)}'
        print(msg)
        logger.debug(msg)

        # Save model
        if e % 5 == 0:
            torch.save({
                'epoch': e,
                'model_color': model_color.state_dict(),
                'optimizer_color': optimizer_color.state_dict(),
                'model_type': model_type.state_dict(),
                'optimizer_type': optimizer_type.state_dict()
            }, f'ckpts/{log_file}_epoch{e}.pt')

if __name__ == '__main__':
    print(f'running script {__file__}')

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=True) # 'config.json'
    parser.add_argument('-p', '--pretrained', default=None)
    args = parser.parse_args()

    train(args=args)