import json
from json import JSONEncoder
import numpy as np

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def main():
    numpyArrayOne = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
    numpyArrayTwo = np.array([[51, 61, 91], [121, 118, 127]])

    # Serialization
    numpyData = {"arrayOne": numpyArrayOne, "arrayTwo": numpyArrayTwo}
    print("serialize NumPy array into JSON and write into a file")
    with open("numpyData.json", "w") as write_file:
        json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
    print("Done writing serialized NumPy array into file")

    # Deserialization
    print("Started Reading JSON file")
    with open("numpyData.json", "r") as read_file:
        print("Converting JSON encoded data into Numpy array")
        decodedArray = json.load(read_file)

        finalNumpyArrayOne = np.asarray(decodedArray["arrayOne"])
        print("NumPy Array One")
        print(finalNumpyArrayOne)
        finalNumpyArrayTwo = np.asarray(decodedArray["arrayTwo"])
        print("NumPy Array Two")
        print(finalNumpyArrayTwo)

if __name__ == '__main__':
    main()