#!/usr/bin/python

import jetson.inference
import jetson.utils
import argparse
#Note these Jetson modules where installed during the 'sudo make install' step of 'jetson-inference' repo

#boilerplate code to parse the image filename and optional --network parameter
#parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be: googlenet, resnet-18, etc. (see --help for others)")
opt = parser.parse_args()

#use script as follows:
#./JetsonNano_Recognition.py my_image.jpg (default if using googlenet)
#./JetsonNano_Recognition.py --network=resnet-18 my_image.jpg (if using ResNet-18 network or specify any other already installed)

#loading image from disk
img, width, height = jetson.utils.loadImageRGBA(opt.filename)

#load the recognition network
net=jetson.inference.imageNet(opt.network)

#classifying the image
class_idx,confidence = net.Classify(img, width, height)

#interpreting the results
#find object description
class_desc = net.GetClassDesc(class_idx)

#print out the result
print("image is recognised as '{:s}' (class#{:d} with {:f}% confidence)".format(class_desc, class_idx, confidence * 100))

