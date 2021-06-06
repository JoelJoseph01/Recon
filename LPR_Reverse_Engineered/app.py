import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder

import glob

import time
from PIL import Image


# Loads model <wpod-net.json> into <wpod-net.h5>
def load_model(path):
    path = splitext(path)[0]
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights('%s.h5' % path)
    print("Loading model successfully...")
    return model

# Preprocess image by image
def preprocess_image(img,resize=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

# Preprocess image by image_path
def Get_image_by_path(image_path,resize=False):
    img = cv2.imread(image_path)
    return img

# forward image through model and return plate's image and coordinates
def get_plate(image, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

class ImageInput:
    # Sample image (for Debugging)
    def sampleImageSingleLP():
        image_path = "IndianLP/car.jpg"
        image = Get_image_by_path(image_path)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        return image

    def sampleImageMultipleLP():
        image_path = "IndianLP/multiple_plates.png"
        img = Get_image_by_path(image_path)
        return img

    def camImage():
        # Preparing cam
        print("Taking snap")
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise Exception("Could not open video device")

        # Capturing image. ret === True on success
        ret, cam_image = video_capture.read()
        video_capture.release()
        cam_imageRGB = cam_image[:,:,::-1]
        return cam_imageRGB


class LPImageExtraction:
    def singleLP(img):
        # Obtain plate image and its coordinates from an image
        try:
            [LpImg],cor = get_plate(img)
            return True, [LpImg]
        except:
            print('No License plate found')
        return False, []


    def multipleLP(img):
        try:
            LpImgs,cor = get_plate(img)
            return True, LpImgs
        except:
            print('No License plate found')
        return False, []

    def loopLP(img):
        try:
            [LpImg],cor = get_plate(img)
            return True, [LpImg]
        except:
            print('No License plate found')
            pass
        return False, []

class ImageLoop:
    def check(image_path):
        # image_path = glob.glob('C:/Users/Joel Joseph/Desktop/Project/Recon/LPR_Reverse_Engineered/NumberPlates/*.jpg')
        # print(image_path)
        image = Get_image_by_path(image_path)
        return image

class Segmentation:
    def apply(lp_img):
        # img_set[[plate_image, thre_mor, binary], [...] ]
        img_set = Segmentation.prepareForSegmentation(lp_img)

        segmented_img = []
        for img in img_set:
            cont, _  = cv2.findContours(img[2], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            test_roi = img[0].copy()
            crop_characters = []
            # Width and Height of character
            digit_w, digit_h = 30, 60

            for c in Segmentation.sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                # Only select contour with defined ratio
                if 1<=ratio<=4:
                    # Select contour which has the height larger than 50% of the plate
                    if h/img[0].shape[0]>=0.5:
                        # Draw bounding box around digit number
                        rect=cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                        cv2.imshow('rect',rect)
                        cv2.waitKey(0)
                        # Separate number and gibe prediction
                        curr_num = img[1][y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)

            print("Detected {} letters...".format(len(crop_characters)))
            fig = plt.figure(figsize=(10,6))
            segmented_img.append(crop_characters)

        return segmented_img



    # BGR --> GRAY --> Blur --> Binary(inverse) --> Dilation
    def prepareForSegmentation(lp_img):
        result = []
        for x in lp_img:
            # Scales, calculates absolute values, and converts the result to 8-bit.
            plate_image = cv2.convertScaleAbs(lp_img[0], alpha=(255.0))
            # Converting to grayscale and blurring the image
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray',gray)
            cv2.waitKey(0)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            # Appling inversed thresh_binary
            binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            ## Applied dilation 
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            result.append([plate_image, thre_mor, binary])
        return result


    # Grabs contour of each digit from left to right
    def sort_contours(cnts,reverse = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts

class OCR:
    def run(chars):
        fig = plt.figure(figsize=(15,3))
        cols = len(chars)
        grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

        final_string = ''
        for i,character in enumerate(chars):
            fig.add_subplot(grid[i])
            title = np.array2string(OCR.predict_from_model(character,model,labels))
            plt.title('{}'.format(title.strip("'[]"),fontsize=20))
            final_string+=title.strip("'[]")

        return final_string

    # pre-processing input images and pedict with model
    def predict_from_model(image,model,labels):
        image = cv2.resize(image,(80,80))
        image = np.stack((image,)*3, axis=-1)
        prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
        return prediction


class Execute:
    def camLoop():
        while True:
            img = ImageInput.camImage()
            is_images, lp_img = LPImageExtraction.singleLP(img)
            if is_images:
                chars_img_list = Segmentation.apply(lp_img)

                lp_num = []
                for chars in chars_img_list:
                    number = OCR.run(chars)
                    print(number)
                    lp_num.append(number)
            else:
                print('No LP detected')


    def sampleSingleLP():
        img = ImageInput.sampleImageSingleLP()
        is_images, lp_img = LPImageExtraction.singleLP(img)
        if is_images:
            chars_img_list = Segmentation.apply(lp_img)

            lp_num = []
            for chars in chars_img_list:
                number = OCR.run(chars)
                print(number)
                lp_num.append(number)
        else:
            print('No LP detected')


    def sampleMultipleLP():
        img = ImageInput.sampleImageMultipleLP()
        is_images, lp_img = LPImageExtraction.multipleLP(img)
        if is_images:
            chars_img_list = Segmentation.apply(lp_img)

            lp_num = []
            for chars in chars_img_list:
                number = OCR.run(chars)
                print(number)
                lp_num.append(number)
        else:
            print('No LP detected')

    def predcheck():
        path=glob.glob('NumberPlates/*.jpg')
        accuracy=81
        for i in path:
            print(i)
            img = ImageLoop.check(i)
            is_images, lp_img = LPImageExtraction.loopLP(img)
            if is_images:
                chars_img_list = Segmentation.apply(lp_img)

                lp_num = []
                for chars in chars_img_list:
                    number = OCR.run(chars)
                    print(number)
                    lp_num.append(number)
                base=os.path.basename(i)
                val=os.path.splitext(base)[0]
                if(val==number):
                    accuracy=accuracy
                else:
                    accuracy-=1

            else:
                print('No LP detected')
                accuracy-=1

        print(accuracy)
# Preparing pretrained model for vehicle recognition
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

# Preparing pretrained model for vehicle recognition
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")


if __name__ == '__main__':
    Execute.sampleSingleLP()
