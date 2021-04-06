
import os
import cv2
import argparse
import json
import math
import numpy as np
import csv

#################################################### keywords
# straight
# straight down
# straight up
# down
# turns left
# turns right
# a right turn
# up a hill
# turning left
# turning right
# going right
# moving forward
# left-turning
# running down
# runs down
# sratight down the street, avenue or road.. highway...
# stops
# is stopped
# wait
# not stopping
# slows down
# speeds up
# speed down
# switches lane to left
# switch left
# switches to the left lane
# switches from the left to the right lane
# switch lane to right afterward
# u-turn
# straight up a low hill
####################################################

keyword_right = [
    'turn right',
    'turns right',
    'turning right',
    'going right',
    'right-turning',
]

keyword_left = [
    'turn left',
    'turns left',
    'turning left',
    'going left',
    'right-turning',
]

keyword_up = [
    'speed up',
    'speeds up',
]

keyword_down = [
    'slow down',
    'slows down',
    'speed down',
    'speeds down',
]

keyword_stop = [ 
    'wait',
    'stops',
    'is stopped',
]

def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path

def is_right_turn(nl):
    for keyword in keyword_right:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def is_left_turn(nl):
    for keyword in keyword_left:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def is_speed_up(nl):
    for keyword in keyword_up:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def is_speed_down(nl):
    for keyword in keyword_down:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def is_stop_motion(nl):
    for keyword in keyword_stop:
        if nl.lower().find(keyword) != -1:
            return True
    return False

def main(args):
    with open(f"data/{args.dataset}-tracks.json",'r') as file:
        track_data = json.load(file)
    with open(f"data/{args.dataset}-motions.json",'r') as file:
        motion_data = json.load(file)

    result = []
    track_list = list(track_data.keys())
    for track_id in track_list:
        nls = track_data[track_id]['nl']
        turn = motion_data[track_id]['turn']
        is_stop = motion_data[track_id]['is_stop']
        is_down = motion_data[track_id]['is_down']

        for nl in nls:
            nl_right = is_right_turn(nl)
            nl_left = is_left_turn(nl)
            nl_down = is_speed_down(nl)
            nl_stop = is_stop_motion(nl)
            value = [track_id, nl, turn, nl_right, nl_left, nl_stop, nl_down]
            result.append(value)

    with open(f'data/{args.dataset}-motions.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvhead = ['track_id', 'nl', 'turn', 'nl_right', 'nl_left', 'nl_stop', 'nl_down']
        csvwriter.writerow(csvhead)
        csvwriter.writerows(result)

if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Calculate motion vector from bbox')
    parser.add_argument('--data_root', dest='data_root', default='./',
                        help='dataset root path')
    parser.add_argument('--dataset', dest='dataset', default='train',
                        help='train or test')

    args = parser.parse_args()

    main(args)