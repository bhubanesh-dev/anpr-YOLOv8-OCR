from functools import cmp_to_key
import os

import cv2
import numpy as np
from deskew import determine_skew
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

from essentials import rotate, show_img, save_img, mk_title

count = 1

def get_license_plate(stock_image):
    # Load the YOLOv8 Model
    model = YOLO('../weights/best.pt')
    # Apply YOLOv8 model predictions and show the license plate
    results = model(stock_image)

    # Get the bounding box from YOLOv8 Model Prediction results
    box = results[0].boxes
    bbox = box.xyxy
    bbox = bbox.numpy()

    if bbox.size == 0:  # Means no detection was made
        return False
    else:
        return results, bbox


def crop_plate(stock_image, results, bbox, op_path=None, org_title=None):
    # Show the successful detection results
    global count

    plotted = results[0].plot(conf=True)
    title = str(count) + '_YOLOv8 License Plate Detection'
    count += 1
    show_img(title, plotted)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, plotted)

    # Store the XY Coordinates in integer format
    x1 = int(bbox[0, 0])
    y1 = int(bbox[0, 1])
    x2 = int(bbox[0, 2])
    y2 = int(bbox[0, 3])

    # Now crop the license plate from the stock image
    cropped = stock_image[y1:y2, x1:x2]
    return cropped


def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((13, 13), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 35)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremove = cv2.merge(result_norm_planes)
    return shadowremove


def preprocess_lPlate(image, op_path=None, org_title=None):
    global count

    # Show the Cropped License Plate
    title = str(count) + "_Cropped License Plate"
    count += 1
    show_img(title, image)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, image)

    # Deskew
    angle = determine_skew(image)
    rotated = rotate(image, angle, (0, 0, 0))
    title = str(count) + "_DeSkewed"
    count += 1
    show_img(title, rotated)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, rotated)

    # Remove shadows for cleaning license plate
    clear = shadow_remove(rotated)
    title = str(count) + "_Clear Image"
    count += 1
    show_img(title, clear)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, clear)

    # Convert to gray scale
    gray = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
    title = str(count) + "_Gray Scale"
    count += 1
    show_img(title, gray)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, gray)

    # Denoise
    den = cv2.bilateralFilter(gray, 21, 75, 75)
    title = str(count) + "_Denoised Image"
    count += 1
    show_img(title, den)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, den)

    # Threshold using Otsu's Thresholding
    thresh = cv2.threshold(den, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    title = str(count) + "_Otsu's Thresholding"
    count += 1
    show_img(title, thresh)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, thresh)

    # Erosion & Dilation
    img_erode = cv2.erode(thresh, (3, 3))
    img_dilate = cv2.dilate(img_erode, (3, 3))
    title = str(count) + "_Erosion & Dilation"
    count += 1
    show_img(title, img_dilate)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, img_dilate)

    return img_dilate, gray


def clean_license_plate(lic_plate, shape, op_path=None, org_title=None):
    global count

    # apply connected component analysis to the threshold image
    output = cv2.connectedComponentsWithStats(lic_plate, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    # initialize an output mask to store all characters parsed from
    # the license plate
    mask = np.zeros(shape, dtype="uint8")
    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    for i in range(1, numLabels):
        # extract the connected component statistics for the current
        # label
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # print(w, h, area)

        # ensure the width, height, and area are all neither too small
        # nor too big
        keep_width = 20 < w < 60
        keep_height = 75 < h < 115
        keep_area = 1250 < area < 4000
        # ensure the connected component we are examining passes all
        # three tests
        if all((keep_width, keep_height, keep_area)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask
            component_mask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, component_mask)
            # show_img("mask", mask)

    title = str(count) + "_Clean License Plate"
    count += 1
    show_img(title, mask)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, mask)

    return mask


def segment_lic_plate(image, op_path=None, org_title=None):
    global count

    # Find contours and get bounding box for each contour
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Sort the bounding boxes from left to right, top to bottom
    # sort by Y first, and then sort by X if Ys are similar
    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]

    bounding_boxes = sorted(bounding_boxes, key=cmp_to_key(compare))
    # Draw bounding boxes
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in range(len(bounding_boxes)):
        x, y, w, h = bounding_boxes[i]
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    title = str(count) + "_Character Segmentation"
    count += 1
    show_img(title, color_img)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, color_img)

    return image, color_img, bounding_boxes


def apply_ocr(image, bounding_boxes, op_path=None, org_title=None):
    global count

    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Define constants
    TARGET_WIDTH = 128
    TARGET_HEIGHT = 128

    chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    # Load the pre-trained convolutional neural network
    model = load_model('../weights/characters_model.weights', compile=False)

    vehicle_plate_number = ""
    i = 1  # to keep track of number of characters
    # Loop over the bounding boxes
    for rect in bounding_boxes:
        # Get the coordinates from the bounding box
        x, y, w, h = rect

        # Crop the character from the mask
        # and apply bitwise_not because in our training data for pre-trained model
        # the characters are black on a white background
        crop = image[y:y + h, x:x + w]
        crop = cv2.bitwise_not(crop)

        # Get the number of rows and columns for each cropped image
        # and calculate the padding to match the image input of pre-trained model
        rows = crop.shape[0]
        columns = crop.shape[1]
        paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

        # Apply padding to make the image fit for neural network model
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        # Convert and resize image
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

        crop_path = op_path + "/Cropped Characters"
        if not os.path.exists(crop_path):
            os.makedirs(crop_path)
        title = "Cropped Characters_" + str(i)
        show_img(title, crop)
        output_txt = mk_title(org_title, title)
        save_img(crop_path, output_txt, crop)
        i += 1

        # Prepare data for prediction
        crop = crop.astype("float") / 255.0
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)

        # Make prediction
        prob = model.predict(crop)[0]
        idx = np.argsort(prob)[-1]
        vehicle_plate_number += chars[idx]

        # Show bounding box and prediction on image
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_img, chars[idx], (x - 15, y), font, 0.8, (0, 0, 255), 2)

    title = str(count) + "_Character Recognitions"
    count = 1
    show_img(title, color_img)
    output_txt = mk_title(org_title, title)
    save_img(op_path, output_txt, color_img)

    print("Vehicle Plate Number: " + vehicle_plate_number)
    return color_img, vehicle_plate_number
