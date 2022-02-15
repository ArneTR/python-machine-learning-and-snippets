# Pre-Crop
# convert bird2.jpg -crop 530x530+230+400 bird2-cropped.jpg


# import the necessary packages
import pytesseract
import argparse
import cv2
import numpy as np
# construct the argument parser and parse the arguments}
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd")
args = vars(ap.parse_args())


# load the input image and convert it from BGR to RGB channel
# ordering}
image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.bitwise_not(img_bin)

kernel = np.ones((2, 1), np.uint8)
img = cv2.erode(gray, kernel, iterations=1)
img = cv2.dilate(img, kernel, iterations=1)


# use Tesseract to OCR the image
text = pytesseract.image_to_string(image, lang="deu")
#print(pytesseract.get_languages())
print(text)


# Easier: tesseract bird2-cropped.jpg stdout -l deu
