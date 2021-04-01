import json
import os
import re
from stop_words import ENGLISH_STOP_WORDS as ESW
from car_colors import COLORS
from car_types import TYPES

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

if __name__ == '__main__':
    main()

