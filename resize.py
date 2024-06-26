#!/usr/bin/env python3

import cv2
import os
import argparse


def resizeImg(imgPath, outFolder, size):
    img = cv2.imread(imgPath)
    if img is None:
        print(f"Error trying to load {imgPath}")
        return

    resized = cv2.resize(img, size)
    outPath = os.path.join(outFolder, os.path.basename(imgPath))
    cv2.imwrite(outPath, resized)
    print(f"Image {imgPath} resized")


def resizeImgInFolder(inFolder, outFolder, size):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    for root, _, files in os.walk(inFolder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                imgPath = os.path.join(root, file)
                resizeImg(imgPath, outFolder, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images")
    parser.add_argument("input_folder", help="Images path")
    parser.add_argument("output_folder", help="Output path")
    args = parser.parse_args()

    width = 300
    height = 500

    size = (width, height)
    resizeImgInFolder(args.input_folder, args.output_folder, size)
