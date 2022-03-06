##################################################
##     ##     ##    ## ### ##     ##    ###     ##
#### #### ### ## ## ##  ## ## ### ## ### ## ### ##
#### #### ### ##   ### # # ## ### ## ### ## ### ##
#### #### ### ## ## ## ##  ##     ## ### ## ### ##
#### ####     ## ## ## ### ## ### ##    ###     ##
##################################################

## This is UDACITY tutorial for object detection from a photograph, with more comments and several additions
# https://www.udacity.com/blog/2021/06/tensorflow-object-detection.html
# for "draw_bounding_box_on_image" and "draw_boxes fonctuions":
# https://www.tensorflow.org/hub/tutorials/object_detection

# Mandatory imports

# Following 2 lines suppress warnings and info output from tensorflow, 
# see more explanation from https://programmersought.com/article/99836021767/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf # Main library
import tensorflow_hub as hub # hub for reusable tensorflow models https://githubhelp.com/tensorflow/hub
import numpy as np # Numerical operations in python, for several size / transformation operations

# We'll use requests and PIL to download and process images
import requests # retrieving even online images  , see "load_image" function below
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def load_image(image_url): # enter the URL of your image as a string to "image_url" 
    img = Image.open(requests.get(image_url, stream=True).raw) # retrieve the image
    return img
    
    
def resize_image(img):
    maxsize = (1024, 1024) # Resize image to be no larger than 1024x1024 pixels
    img.thumbnail(maxsize, Image.ANTIALIAS) # ANTIALIAS keeps aspect ratio of image the same
    return img

# Draw the boxes

def draw_bounding_box_on_image(image,ymin,xmin,ymax,xmax,color,font,thickness=4,display_str_list=()):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    # following two lines are customizations for boxes and implementation
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),(left, top)],width=thickness,fill=color)
    # down to for loop, commands indicate the locations of texts, i.e. where to put labels of boxes
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
    # implementing what was customized above with occassional hard-coded constants
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),(left + text_width, text_bottom)], fill=color)
    draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
    text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=4, min_score=0.1):
    colors = list(ImageColor.colormap.values())
    try: # The string below might be changed, i was satisfied with default font.
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()
    for i in range(min(boxes.shape[0], max_boxes)):  # max boxes to draw according to the set rules below
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i]) # borders of boxes
        if scores[i] < 0.9: # additional OPTIONAL filter to remove obvious classifications
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            # see the function "draw_bounding_box_on_image" above
            draw_bounding_box_on_image(image_pil,ymin,xmin,ymax,xmax,color,font,display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image

# Here, tutorial employed Faster Regional Convolutional Neural Network with 
# Inception ResNet image classification architecture of Google.
# The following three lines are only parts requiring internet connection
module_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_url).signatures['default'] # Loading aforementioned module, in case of a problem, download the module in tar.gz from 
                                                      # tensorflow website manually and put it in the error creatign temp folder and extract 
                                                      # till seeing .pb file and other folders
image_url = "https://farm1.staticflickr.com/6188/6082105629_db7abe41b9_o.jpg"

# Loading and resizing
img = load_image(image_url)
img = resize_image(img)

# Checking if everything is correct in image loading part
numpy_img = np.asarray(img)
print(numpy_img.shape)

# Preprocessing
converted_img  = tf.image.convert_image_dtype(img, tf.float32) # Converting to native tensorflow datatype
scaled_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...] # scaling of 255 to between 0-1

output = detector(scaled_img) # The actual RCNN processing

# The list of objects detected with a specific confidence are printed with the following for loop
for obj, confidence in list(zip(output['detection_class_entities'], output['detection_scores']))[:10]:  # the [:10] gives the 10 highest confidence detections
 print("Detected {} with {:.2f}% confidence".format(obj, confidence))

# Finally, drawing boxes on the image we supplied with labels and confidence intervals (a.k.a. scores)
img2 = draw_boxes(np.array(img), output["detection_boxes"], output["detection_class_entities"].numpy(), output["detection_scores"])

# Convert the image back into PIL.Image so we can display it
img2 = Image.fromarray(img2)

img2.show() # Shows the image with boxes, labels, scores
