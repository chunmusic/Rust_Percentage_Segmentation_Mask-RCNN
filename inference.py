import os
import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

import itertools

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_dir):
    
    model = tf.saved_model.load(str(model_dir))
    return model
    

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/pongsakorn/kaggle_ws/rust_segmentation_ws/dataset/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def run_inference_for_single_image(model, image_np):
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop("num_detections"))

    detections = dict(itertools.islice(detections.items(), num_detections))

    detections["num_detections"] = num_detections


    # Handle models with masks:
    if "detection_masks" in detections:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              detections["detection_masks"][0], detections["detection_boxes"][0],
               image_np.shape[0], image_np.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        detections["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return detections


def show_inference(model, image_np):

    # Load image
    # image_np = np.array(Image.open(image_path))

    image_np_with_detections = image_np.copy()

    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)

    boxes = np.asarray(output_dict["detection_boxes"][0])
    classes = np.asarray(output_dict["detection_classes"][0]).astype(np.int64)
    scores = np.asarray(output_dict["detection_scores"][0])
    mask = np.asarray(output_dict["detection_masks_reframed"])


    #print(np.reshape(mask, (-1, mask.shape[-1])).astype(np.float32).sum())

    # Visualizing the results
    final_img,rust_ret,rust_percent = vis_util.visualize_boxes_and_labels_on_image_array(
                                        image_np_with_detections,
                                        boxes,
                                        classes,
                                        scores,
                                        category_index,
                                        instance_masks=mask,
                                        use_normalized_coordinates=True,
                                        line_thickness=3)
                            
    
    return final_img,rust_ret,rust_percent


model_dir = '/home/pongsakorn/kaggle_ws/rust_segmentation_ws/output/export_saved_model/saved_model'
masking_model = load_model(model_dir)


cap = cv2.VideoCapture("rust.mp4")

while cap.isOpened():
    ret, img = cap.read()

    if ret == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_img,rust_ret,rust_percent = show_inference(masking_model, img)
        
        if rust_ret:
            cv2.putText(final_img,"Rust Percent",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(112,40,9),3)
            cv2.putText(final_img,str(round(rust_percent,2))+" %",(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(112,40,9),3)

        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

        cv2.imshow("img",final_img)

        # Image.fromarray(final_img).show() # For running on python

    else:
        break;


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
