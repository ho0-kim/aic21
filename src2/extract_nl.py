import json
import os
import re
from stop_words import ENGLISH_STOP_WORDS as ESW

def tokenize(sentence):
    tokens = []
    word = re.sub(r'[^a-zA-Z ]', '', sentence).split()
    for w in word:
        w = w.lower()
        if w not in ESW:
            tokens.append(w)
    return tokens

def main():
    file_path = os.path.join('../../data/train-tracks.json')

    with open(file_path, 'rt') as f:
        json_data = json.load(f)

    nls = {}
    for k, v in json_data.items():
        nl = v['nl']
        nls.update({k: nl})
        for s in nl:
            tokens = tokenize(s)
            print(s)
            print(tokens)
    #print(nls)

if __name__ == '__main__':
    main()

