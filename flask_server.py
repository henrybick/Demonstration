from flask import Flask, request, send_file, render_template, send_from_directory
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

UPLOAD_FOLDER = os.path.basename('uploads')
RGB_UPLOAD_FOLDER = os.path.basename('input-images-uploads')
MASK_UPLOAD_FOLDER = os.path.basename('mask-images-uploads')

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RGB_UPLOAD_FOLDER'] = RGB_UPLOAD_FOLDER
app.config['MASK_UPLOAD_FOLDER'] = MASK_UPLOAD_FOLDER
 
@app.route('/')
def titlepage():
    return render_template('titlepage.html')

@app.route('/annotator/')
def annotator():
    return render_template('annotator.html')

@app.route('/annotatorcode/')
def annotatorcode():
    return render_template('annotatorcode.html')

@app.route('/background/')
def input_background():
    return render_template('input.html')

@app.route('/nbackground/')
def input_nb():
    return render_template('nbinput.html')

@app.route('/binarymask/')
def input_binarymask():
    return render_template('bminput.html')

@app.route('/uploaddataset' , methods =['POST', 'GET'])
def upload_input_image_set():
    return render_template('inputimageset.html')

@app.route('/result', methods = ['POST', 'GET'])
def remove_background():

    if request.method == 'POST':
        imagefile = request.files['image']
        csvfile = request.files['csv']

        i = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(i)

        c = os.path.join(app.config['UPLOAD_FOLDER'], csvfile.filename)
        csvfile.save(c)

        print("Files uploaded successfully")

    #Reading in the initial colour image
    img, path, filename = pcv.readimage(i)

    #Converting the CSV file to a txt file
    convert_csv(csvfile.filename)

    fn = csvfile.filename.split(".")[0]
    plotstextfilename = fn + ".txt"

    #Reading in the converted CSV file into an array of plotted points
    pts = np.array(get_values(plotstextfilename))

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

    #Converting the image from a Numpy array to a Base64 string to allow the website to render it properly
    im = Image.fromarray(out.astype("uint8"))
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)
    outimage = base64.b64encode(rawBytes.read())

    #Returning the template and image
    return render_template("result.html", image = outimage)

@app.route('/nbresult', methods = ['POST', 'GET'])
def naive_bayes():

    if request.method == 'POST':
        imagefile = request.files['image']
        pdffile = request.files['pdf']

        i = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(i)

        p = os.path.join(app.config['UPLOAD_FOLDER'], pdffile.filename)
        pdffile.save(p)

        print("Files uploaded successfully")

    #Reading in the image
    img, path, filename = pcv.readimage(i)

    #Creating the mask from the base image and the model
    device, mask = pcv.naive_bayes_classifier(img, pdf_file=pdffile.filename, device=0, debug="print")
    
    #Applying the mask to the colour image
    device, masked_image = pcv.apply_mask(img, mask['plant'], 'white', device, debug="print")

    #Converting the image from a Numpy array to a Base64 string to allow the website to render it properly
    im = Image.fromarray(masked_image.astype("uint8"))
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)
    outimage = base64.b64encode(rawBytes.read())

    #Returning the template and image
    return render_template("result.html", image = outimage)

@app.route('/bmresult', methods =['POST', 'GET'])
def binary_mask():

    if request.method == 'POST':
        imagefile = request.files['image']

        i = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(i)

        print("Files uploaded successfully")

    #Reading in the image
    img, path, filename = pcv.readimage(i)

    #Defining easier to use names
    names = {"h": "hue", "s": "saturation", "v": "value"}

    #Converting the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Spliting the h, s, v values up from eachother
    h, s, v = cv2.split(hsv)

    #Defining the channels for easier use
    channels = {"h": h, "s": s, "v": v}

    s = channels["s"]

    #Creating the binary threshold image using the channel, the threshold, the max value, and the object type
    device, s_thresh = pcv.binary_threshold(s, 85, 255, 'light', device=0, debug=None)

    #Converting the image from Numpy array to Base64 string. The image is not BGR so it does not need to be converted to RGB this time
    im = Image.fromarray(s_thresh.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)
    outimage = base64.b64encode(rawBytes.read())

    #Returning the template and image
    return render_template("result.html", image = outimage)

@app.route('/inputimageresult' , methods =['POST', 'GET'])
def imageset_result():
    if request.method == 'POST':

        uploaded_files_rgb = request.files.getlist("image")
        uploaded_files_mask = request.files.getlist("maskimage")

        for x in uploaded_files_rgb:
            i = os.path.join(app.config['RGB_UPLOAD_FOLDER'], x.filename)
            x.save(i)

        for x in uploaded_files_mask:
            m = os.path.join(app.config['MASK_UPLOAD_FOLDER'], x.filename)
            x.save(m)

    return "Files uploaded successfully"

@app.route('/nbtrain', methods =['POST', 'GET'])
def nbtrain():
    inputimages = "./uploads/input-images"
    maskimages = "./uploads/mask-images"
    outfile = "testpdf.txt"
    naive_bayes_train(inputimages, maskimages, outfile)

    return send_from_directory(directory=UPLOAD_FOLDER, filename=outfile, as_attachment=True)

@app.route('/multitrain', methods =['POST', 'GET'])
def multitrain():

    if request.method == 'POST':
        imagefile = request.files['image']
        jsonfile = request.files['json']

        i = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(i)

        j = os.path.join(app.config['UPLOAD_FOLDER'], jsonfile.filename)
        jsonfile.save(j)

        print("Files uploaded successfully")

    inputimage = i
    inputjson = j
    outfile = "multitestpdf.txt"

    inputpdf = convert_json(inputjson)

    naive_bayes_multiclass(inputpdf, outfile)

    return send_from_directory(directory=UPLOAD_FOLDER, filename=outfile, as_attachment=True)

def get_values(filename):
    with open(filename) as f:
        l = f.readlines()
    return [tuple(map(int, x.replace('\n', '').split(', '))) for x in l[1:]]

def convert_csv(csv_filename):
    rootDir ="."
    for root, dir, files in os.walk(rootDir):
        for name in files:
            #print('files\t%s' % name)
            if fnmatch.fnmatch(name, csv_filename):
                stem = []
                name = os.path.join(root, name)
                file = open(name,'r')
                try:
                    reader = csv.DictReader(file)
                    for row in reader:
                        val= ast.literal_eval(row['region_shape_attributes'])
                        x_coordinates = val["all_points_x"]
                        y_coordinates = val["all_points_y"]
                        for x,y in zip(x_coordinates,y_coordinates):
                            stem.append(str(x) + ", " + str(y))
                        
                finally:
                    file.close()
                
                name = name[:-4]
                stem_file_name =name+ ".txt"    
                file =open(stem_file_name,"w")
                file.write("x,y\n")
                stem = list(OrderedDict.fromkeys(stem))
                for s in stem:
                    # print(c[0], c[1])
                    file.write(s+'\n')
                file.close()
                #print (stem)

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

    return output_tsv(rgbdict)
    
def output_tsv(*dicts):
    filename = 'output.txt'
    title = []
    lists = []
    for d in dicts:
        for k, v in d.items():
            # Add the keys of our dicts as titles, and our lists as a nested list
            title.append(k)
            lists.append(v)
    # Transpose our lists, zip_longest adds None if no other elements can be found, perfect for us
    lists = list(map(list, izip_longest(*lists)))
    with open(filename, 'wb') as output:
        tsv_output = csv.writer(output, delimiter='\t')
        tsv_output.writerow(title)
        for r in lists:
            tsv_output.writerow(r)
    return filename

#===========================================================================================================================#
#===========================================================================================================================#
#===========================================================================================================================#
#===========================================================================================================================#
#===========================================================================================================================#
#===========================================================================================================================#


def naive_bayes_train(imgdir, maskdir, outfile, mkplots=False):

    # Initialize color channel ndarrays for plant (foreground) and background
    plant = {"hue": np.array([], dtype=np.uint8), "saturation": np.array([], dtype=np.uint8),
             "value": np.array([], dtype=np.uint8)}
    background = {"hue": np.array([], dtype=np.uint8), "saturation": np.array([], dtype=np.uint8),
                  "value": np.array([], dtype=np.uint8)}

    # Walk through the image directory
    print("Reading images...")
    for (dirpath, dirnames, filenames) in os.walk(imgdir):
        for filename in filenames:
            # Is this an image type we can work with?
            if filename[-3:] in ['png', 'jpg', 'jpeg']:
                # Does the mask exist?
                if os.path.exists(os.path.join(maskdir, filename)):
                    # Read the image as BGR
                    img = cv2.imread(os.path.join(dirpath, filename), 1)
                    # Read the mask as grayscale
                    mask = cv2.imread(os.path.join(maskdir, filename), 0)

                    # Convert the image to HSV and split into component channels
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hue, saturation, value = cv2.split(hsv)

                    # Store channels in a dictionary
                    channels = {"hue": hue, "saturation": saturation, "value": value}

                    # Split channels into plant and non-plant signal
                    for channel in channels.keys():
                        fg, bg = _split_plant_background_signal(channels[channel], mask)

                        # Randomly sample from the plant class (sample 10% of the pixels)
                        fg = fg[np.random.random_integers(0, len(fg) - 1, int(len(fg) / 10))]
                        # Randomly sample from the background class the same n as the plant class
                        bg = bg[np.random.random_integers(0, len(bg) - 1, len(fg))]
                        plant[channel] = np.append(plant[channel], fg)
                        background[channel] = np.append(background[channel], bg)

    # Calculate a probability density function for each channel using a Gaussian kernel density estimator
    # Create an output file for the PDFs
    out = open(outfile, "w")
    out.write("class\tchannel\t" + "\t".join(map(str, range(0, 256))) + "\n")
    for channel in plant.keys():
        print("Calculating PDF for the " + channel + " channel...")
        plant_kde = stats.gaussian_kde(plant[channel])
        bg_kde = stats.gaussian_kde(background[channel])
        # Calculate p from the PDFs for each 8-bit intensity value and save to outfile
        plant_pdf = plant_kde(range(0, 256))
        out.write("plant\t" + channel + "\t" + "\t".join(map(str, plant_pdf)) + "\n")
        bg_pdf = bg_kde(range(0, 256))
        out.write("background\t" + channel + "\t" + "\t".join(map(str, bg_pdf)) + "\n")
        if mkplots:
            # If mkplots is True, make the PDF charts
            _plot_pdf(channel, os.path.dirname(outfile), plant=plant_pdf, background=bg_pdf)

    out.close()

def naive_bayes_multiclass(samples_file, outfile, mkplots=False):

    # Initialize a dictionary to store sampled RGB pixel values for each input class
    sample_points = {}
    # Open the sampled points text file
    f = open(samples_file, "r")
    # Read the first line and use the column headers as class labels
    header = f.readline()
    header = header.rstrip("\n")
    class_list = header.split("\t")
    # Initialize a dictionary for the red, green, and blue channels for each class
    for cls in class_list:
        sample_points[cls] = {"red": [], "green": [], "blue": []}
    # Loop over the rest of the data in the input file
    for row in f:
        # Remove newlines and quotes
        row = row.rstrip("\n")
        row = row.replace('"', '')
        # If this is not a blank line, parse the data
        if len(row) > 0:
            # Split the row into a list of points per class
            points = row.split("\t")
            # For each point per class
            for i, point in enumerate(points):
                if len(point) > 0:
                    # Split the point into red, green, and blue integer values
                    red, green, blue = map(int, point.split(","))
                    # Append each intensity value into the appropriate class list
                    sample_points[class_list[i]]["red"].append(red)
                    sample_points[class_list[i]]["green"].append(green)
                    sample_points[class_list[i]]["blue"].append(blue)
    f.close()
    # Initialize a dictionary to store probability density functions per color channel in HSV colorspace
    pdfs = {"hue": {}, "saturation": {}, "value": {}}
    # For each class
    for cls in class_list:
        # Create a blue, green, red-formatted image ndarray with the class RGB values
        bgr_img = cv2.merge((np.asarray(sample_points[cls]["blue"], dtype=np.uint8),
                             np.asarray(sample_points[cls]["green"], dtype=np.uint8),
                             np.asarray(sample_points[cls]["red"], dtype=np.uint8)))
        # Convert the BGR ndarray to an HSV ndarray
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        # Split the HSV ndarray into the component HSV channels
        hue, saturation, value = cv2.split(hsv_img)
        # Create an HSV channel dictionary that stores the channels as lists (horizontally stacked ndarrays)
        channels = {"hue": np.hstack(hue), "saturation": np.hstack(saturation), "value": np.hstack(value)}
        # For each channel
        for channel in channels.keys():
            # Create a kernel density estimator for the channel values (Guassian kernel)
            kde = stats.gaussian_kde(channels[channel])
            # Use the KDE to calculate a probability density function for the channel
            # Sample at each of the possible 8-bit values
            pdfs[channel][cls] = kde(range(0, 256))
    if mkplots:
        # If mkplots is True, generate a density curve plot per channel for each class
        for channel, cls in pdfs.items():
            _plot_pdf(channel, os.path.dirname(outfile), **cls)
    # Write the PDFs to a text file
    out = open(outfile, "w")
    # Write the column labels
    out.write("class\tchannel\t" + "\t".join(map(str, range(0, 256))) + "\n")
    # For each channel
    for channel, cls in pdfs.items():
        # For each class
        for class_name, pdf in cls.items():
            # Each row is the PDF for the given class and color channel
            out.write(class_name + "\t" + channel + "\t" + "\t".join(map(str, pdf)) + "\n")


def _split_plant_background_signal(channel, mask):
    """Split a single-channel image by foreground and background using a mask

    :param channel: ndarray
    :param mask: ndarray
    :return plant: ndarray
    :return background: ndarray
    """
    plant = channel[np.where(mask == 255)]
    background = channel[np.where(mask == 0)]

    return plant, background


def _plot_pdf(channel, outdir, **kwargs):
    """Plot the probability density function of one or more classes for the given channel

    :param channel: str
    :param outdir: str
    :param kwargs: dict
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    for class_name, pdf in kwargs.items():
        plt.plot(pdf, label=class_name)
    plt.legend(loc="best")
    plt.savefig(os.path.join(outdir, str(channel) + "_pdf.svg"))
    plt.close()

if __name__ == "__main__":
    app.run(debug = True)
