from flask import Flask, request, send_file, render_template
import sys, traceback
import cv2
import numpy as np
import argparse
import string
import plantcv as pcv
import os
import csv
import ast
import fnmatch
from collections import OrderedDict
from scipy.misc import toimage
from base64 import b64encode
import io
import base64
from PIL import Image
from werkzeug.utils import secure_filename  
from scipy import stats 
import json
import PIL
from matplotlib.path import Path
from itertools import izip_longest

def load_json(filename):
    with open(filename) as infile:
        return json.load(infile)

def convert_json(filename):
    rawjson = load_json(filename)
    outputdict = {}
    rgbdict = {}
    device = 0

    # Ignoring the first key of the JSON which will be the filename + filesize 
    key = list(rawjson.keys())[0]
    tempdict = rawjson[key]
    regions = tempdict['regions']

    # Iterating through each key-value pair in the regions dictionary, finding the names of the regions the user has used, finding the coords associated with it
    # and adding it to the dictionary
    for id, values in regions.iteritems():
        region_attributes = values['region_attributes']
        identifier = list(region_attributes.keys())[0]

        name = values['region_attributes'][identifier]

        x_points = values['shape_attributes']['all_points_x']
        y_points = values['shape_attributes']['all_points_y']

        coords = list(zip(x_points, y_points))

        outputdict[name] = coords

    print outputdict

    for z in outputdict:
        rgblist = []
        pts = np.array(outputdict[z])

        img, path, filename = pcv.readimage("ogimage.png")

        #Creating mask arrays of 0's
        mask = np.zeros((img.shape[0], img.shape[1]))

        #Filling the polygon
        cv2.fillConvexPoly(mask, pts, 1)
        mask = mask.astype(np.bool)

        #Creating the output
        out = np.zeros_like(img)
        out[mask] = img[mask]

        #Turning the black background into a white background
        out[np.where((out==[0,0,0]).all(axis=2))] = [255,255,255]

        image = Image.fromarray(out.astype('uint8'), 'RGB')
        width, height = image.size

        for x in range(width):
            for y in range(height):
                r,g,b = image.getpixel((x,y))
                if not(r==g==b==255):  
                    temprgb = str(r) + ", " + str(g) + ", " + str(b)
                    rgblist.append(temprgb)   
        
        rgbdict[z] = rgblist

    pretty_print(rgbdict)

def pretty_print(*dicts):
    title = []
    lists = []
    for d in dicts:
        for k, v in d.items():
            # Add the keys of our dicts as titles, and our lists as a nested list
            title.append(k)
            lists.append(v)
    # Transpose our lists, zip_longest adds None if no other elements can be found, perfect for us
    lists = list(map(list, izip_longest(*lists)))
    with open('output.txt', 'wb') as output:
        tsv_output = csv.writer(output, delimiter='\t')
        tsv_output.writerow(title)
        for r in lists:
            tsv_output.writerow(r)
    return True

if __name__ == "__main__":
    convert_json("data.json")
