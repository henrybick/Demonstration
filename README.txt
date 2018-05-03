Server files for the Annotation Tool extending the VGG VIA tool and using PlantCV:

Server should be run on the same network as the machines that want to access it. 

Python 2.7 is recommended 

NOTE: Google Chrome has some issues with allowing the user to download region data as a CSV or JSON. If this error occurs try
another browser like Firefox.

Manual Installation:
1. Install venv and pip install PlantCV and its dependancies (detailed in requirements.txt)
2. Activate venv and navigate to the server's directory
3. Run the server python file from the command line

Upload folders should be setup in the same configuration as shown on the source code's GitHub:
-Folders can be renamed however the corresponding variables should also be renamed inside the server source code
-Folders must be cleaned manually by the server owner

FEATURE EXAMPLES:
These examples assume the server has been installed correctly and is running.

Annotating an Image:
-Navigate to the annotation tool from the landing page
-Under the nagivation bar select the "Load or Add Images" option
-Select the desired region shape from the side panel and either drag (for conventional shapes) or click for each point (for polygons) to create a region
-Select the region attributes option from the side panel to add labels to the regions
-Under the "Annotation" menu on the navigation bar you can decide to download the region data as either CSV (for cropping) or JSON (for multiclass model training)

Creating a Binary Mask:
-Select the binary mask option from the "Image Processing" menu on the navigation bar
-Upload the desired image
-Right click to save image once the result is displayed

Cropping an Image:
-Annotate an image using the annotation tool with ONE region shape
-Download the region data in the form of a CSV file
-Navigate to the cropping tool from the navigation bar
-Upload the image and the region data CSV file
-Right click to save image once the result is displayed

Errors: Errors can occur if there are multiple regions present in the CSV file.

Uploading a Dataset:
-Choose the upload dataset option from the navigation bar
-Shift click to select multiple images to upload at once

Training a Naive Bayes Model:
-Make sure you have first uploaded both an RGB and mask dataset
-Choose the train a model option from the navigation bar
-Save the output file to desired location

Training a Multiclass Model:
-Begin by annotating an image with multiple regions to highlight features.
-Label the regions to identify them
-Download the region data as a JSON file
-Find the link to multiclass model training under the navigation bar dropdown
-Upload the image and the JSON file
-Save the resulting txt model to the desired location

Using a Model to Remove Background:
-Select the remove background using model option from the navigation bar
-Upload the new RGB image and the trained model
-Right click to save the resulting image




