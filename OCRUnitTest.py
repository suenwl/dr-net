#%%
from PIL import Image
import pytesseract
import pandas as pd
from OCREngine import OCREngine
import numpy as np
import cv2
import os

#%%
def remove_lines(input):
    pil_image = input.convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    img = open_cv_image[:, :, ::-1].copy()
    # Convert to grey for easier processing
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #inverse colour on image for easier processing of lines
    img = cv2.bitwise_not(img)
    th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    #show blackwhite version
    #cv2.imshow("blackwhite_orginal", th2)
    #cv2.imwrite("blackwhite_orginal.jpg", th2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    horizontal = th2
    vertical = th2
    rows,cols = horizontal.shape
    horizontalsize = int(cols / 15)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    #show horizontal lines
    #cv2.imshow("horizontal", horizontal)
    #cv2.imwrite("horizontal.jpg", horizontal)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print('horizontal')

    #inverse the image, so that lines are black for masking
    horizontal_inv = cv2.bitwise_not(horizontal)
    #perform bitwise_and to mask the lines with provided mask
    masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
    #reverse the image back to normal
    masked_img_inv = cv2.bitwise_not(masked_img)
    #show removal of horizontal lines
    #cv2.imshow("masked img", masked_img_inv)
    #cv2.imwrite("result2.jpg", masked_img_inv)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print('masked_img_inv')

    verticalsize = int(rows / 30)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    #show vertical lines
    #cv2.imshow("vertical", vertical)
    #cv2.imwrite("vertical.jpg", vertical)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print('vertical')

    masked_img_inv = cv2.bitwise_not(masked_img_inv)
    #inverse the image, so that lines are black for masking
    vertical_inv = cv2.bitwise_not(vertical)
    #perform bitwise_and to mask the lines with provided mask
    masked_img2 = cv2.bitwise_and(masked_img_inv, masked_img_inv, mask=vertical_inv)
    #reverse the image back to normal
    masked_img_inv2 = cv2.bitwise_not(masked_img2)

    # Dilation and Erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    dilatedImage = cv2.dilate(masked_img_inv2, kernel, iterations=10)
    dilatedImage = cv2.erode(dilatedImage, kernel, iterations=10)

    # Median Blur
    #dilatedImage = cv2.adaptiveThreshold(cv2.medianBlur(dilatedImage, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    #show final result
    #cv2.imshow("masked img2", masked_img_inv2)
    cv2.imwrite("final_result.jpg", dilatedImage)
    #cv2.imwrite("final_result.jpg", masked_img_inv2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return(dilatedImage)




#%%
print("Starting...")
INVOICE_PATH = "/Users/lxg/Documents/Semester Modules/BT3101 Capstone Project/PDF Invoices/OCR Test.jpg"
imageInvoice = Image.open(INVOICE_PATH)
invoice = remove_lines(imageInvoice)
vanil_OCR_output = pytesseract.image_to_string(invoice)
raw_OCR_output = pytesseract.image_to_string(invoice, config="oem==1, textord_heavy_nr==1, textord_min_linesize==0.25, psm==6")
print(vanil_OCR_output)
print(raw_OCR_output)

with open("VanillaOutput.txt", "w") as text_file:
    print(vanil_OCR_output, file=text_file)

with open("RawOutput.txt", "w") as text_file:
    print(raw_OCR_output, file=text_file)

#%%
