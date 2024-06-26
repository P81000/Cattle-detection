#!/usr/bin/env python3

import os
import shutil
import random

source = "/home/p81000/Codes/src/dataset/images"
train = source + "/train"
test = source + "/test"
val = source + "/val"

files = os.listdir(source)
pairs = {os.path.splitext(f)[0] for f in files}

pairs = list(pairs)
random.shuffle(pairs)

train_split = int(0.7 * len(pairs))
test_split = int(0.2 * len(pairs))

trainPairs = pairs[:train_split]
testPairs = pairs[train_split:train_split + test_split]
valPairs = pairs[train_split + test_split:]


def moveFiles(pairs, outFolder):
    for pair in pairs:
        for ext in [".jpg", ".txt"]:
            filename = pair + ext
            sourcepath = os.path.join(source, filename)
            if os.path.exists(sourcepath):
                shutil.move(sourcepath, os.path.join(outFolder, filename))


moveFiles(trainPairs, train)
moveFiles(testPairs, test)
moveFiles(valPairs, val)

print("Done")
