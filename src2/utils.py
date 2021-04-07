import json
import os
import re
from stop_words import ENGLISH_STOP_WORDS as ESW
from car_colors import COLORS
from car_types import TYPES
from collections import Counter

def tokenize(sentence):
    tokens = []
    sentence = sentence.lower()
    word = re.sub(r'[^a-zA-Z ]', '', sentence).split()

    for w in word:
        w = w.lower()
        #if w not in ESW:
        tokens.append(w)
    return tokens

def hasColors(tokens):
    colors = [item for sublist in COLORS for item in sublist]
    bHas = False
    color = None
    for t in tokens:
        if t in colors:
            bHas = True
            color = t
            break
    return bHas, color

def hasTypes(tokens):
    types = [item for sublist in TYPES for item in sublist]
    bHas = False
    type = None
    for i, t in enumerate(tokens):
        if t in types:
            bHas = True
            if t == 'truck' and tokens[i-1] == 'cargo': type = 'cargo truck'
            else: type = t
            break
    return bHas, type

def getColorLabel(tokens):
    bHas, color = hasColors(tokens)
    label = -1

    if bHas:
        for i, clr in enumerate(COLORS):
            if color in clr:
                label = i
                break
        assert (label < len(COLORS))

    return color, label

def getTypeLabel(tokens):
    bHas, type = hasTypes(tokens)
    label = -1

    if bHas:
        for i, mdl in enumerate(TYPES):
            if type in mdl:
                label = i
                break
        assert (label < len(TYPES))

    return type, label

def getTotalColorLabel():
    return len(COLORS)

def getTotalTypeLabel():
    return len(TYPES)

def getColorList(nls):
    colors = list()
    color_count = 0

    for s in nls:
        tokens = tokenize(s)
        color, color_label = getColorLabel(tokens)
        colors.append(color_label)

        if color_label >= 0:
            color_count += 1

    assert (color_count >= 0)
    return colors

def getTypeList(nls):
    types = list()
    type_count = 0

    for s in nls:
        tokens = tokenize(s)
        type, type_label = getTypeLabel(tokens)
        types.append(type_label)

        if type_label >= 0:
            type_count += 1

    assert (type_count >= 0)
    return types

def main():
    file_path = os.path.join('../../data/train-tracks.json')

    with open(file_path, 'rt') as f:
        json_data = json.load(f)

    nls = {}
    color_stat = [0] * len(COLORS)
    type_stat = [0] * len(TYPES)
    for k, v in json_data.items():
        nl = v['nl']
        nls.update({k: nl})
        no_color_count = 0
        no_type_count = 0
        for s in nl:
            tokens = tokenize(s)

            color, color_label = getColorLabel(tokens)
            color_stat[color_label] += 1

            if color_label < 0:
                no_color_count += 1
            ''''
            if color_label < 0:
                print(s)
            '''

            type, type_label = getTypeLabel(tokens)
            type_stat[type_label] += 1
            if type_label < 0:
                no_type_count += 1
                print(s)

            #print(tokens)
        # if no_color_count == len(nl):
        #     print(nl)
        if no_type_count == len(nl):
            print(nl)
    #print(nls)
    #print(color_stat)
    print(type_stat)

keyword_right = [
    'turn right',
    'turns right',
    'turning right',
    'right turn',
    'takes a right',
    'took a right'
]

keyword_left = [
    'turn left',
    'turns left',
    'turning left',
    'left turn',
    'turned left',
    'takes a left'
]

keyword_up = [
    'speed up',
    'speeds up',
]

keyword_down = [
    'slow',
    'slows',
    'slowly'
    'speed down',
    'speeds down',
]

keyword_stop = [ 
    'wait',
    'stops',
    'stopped',
    'waiting'
]

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

def motion_detection(nls):
    right, left, up, down, stop = 0, 0, 0, 0, 0
    for nl in nls:
        right += 1 if is_right_turn(nl) else 0
        left += 1 if is_left_turn(nl) else 0
        up += 1 if is_speed_up(nl) else 0
        down += 1 if is_speed_down(nl) else 0
        stop += 1 if is_stop_motion(nl) else 0
    return [right, left, up, down, stop]

def motion_calculation(track_id):
    with open("../data/test-motions.json", 'r') as file:
        motion_data = json.load(file)
    turn = motion_data[track_id]['turn']
    down = 1 if motion_data[track_id]['is_down'] else 0
    stop = 1 if motion_data[track_id]['is_stop'] else 0
    right = 1 if turn > 60 else 0 # alternative value: 20
    left = 1 if turn < -60 else 0

    return [right, left, 0, down, stop]

if __name__ == '__main__':
    main()

