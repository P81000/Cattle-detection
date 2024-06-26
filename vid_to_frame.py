#!/usr/bin/env python3

import cv2
import os
import argparse


def extFromVideo(vidPath, outPath, frameStartCount):
    cap = cv2.VideoCapture(vidPath)

    if not cap.isOpened():
        print(f"Error trying to open {vidPath}")
        return frameStartCount

    frameCount = frameStartCount
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frameName = os.path.join(outPath, f"frame_{frameCount:06d}.jpg")
        cv2.imwrite(frameName, frame)
        frameCount += 1

    cap.release()
    print(f"Complete! Frames from {vidPath} saved in {outPath}")


def extFromPath(inPath, outPath):
    frameStartCount = 0

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    for root, _, files in os.walk(inPath):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                vidPath = os.path.join(root, file)
                frameStartCount = extFromVideo(vidPath, outPath, frameStartCount)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames form videos")
    parser.add_argument("input_folder", help="Videos")
    parser.add_argument("output_folder", help="Target")
    args = parser.parse_args()

    extFromPath(args.input_folder, args.output_folder)
