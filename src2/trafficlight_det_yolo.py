import json
import subprocess

tracks = json.load(open('../../data/test-tracks.json', 'r'))
uuids = list(tracks.keys())

out_file = '../../data/test-trafficlight_yolo.json'
out = {}

for i, uuid in enumerate(uuids):
    print(i, uuid)
    output = subprocess.check_output('./darknet detect ' + 
                                    'cfg/yolov3.cfg ' +
                                    'yolov3.weights ' +
                                    f'../../{tracks[uuid]["frames"][0]}', shell=True)
    print('traffic light' in str(output))
    out[uuid] = 'traffic light' in str(output)

json.dump(out, open(out_file, 'w'), indent=4)