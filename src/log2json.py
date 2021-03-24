import json
import os, fnmatch

base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, 'log\\')

listOfFiles = os.listdir(log_dir)
pattern = "*.log"
result_dict = {}

for file in listOfFiles:
    if fnmatch.fnmatch(file, pattern):
        print(file)
        file_path = os.path.join(log_dir, file)
        with open(file_path, 'rt') as f:
            file = file.split('.')[0]
            data = f.read()
            ct = data.split('\n')
            result_dict.update({file: ct})

result_path = os.path.join(log_dir, 'result.json')
with open(result_path, 'wt') as f:
    json_str = json.dumps(result_dict)
    f.write(json_str)

