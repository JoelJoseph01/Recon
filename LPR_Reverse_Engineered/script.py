import cv2
import glob
import os
# from app.py import *
from os.path import splitext,basename
path=glob.glob('Recon\LPR_Reverse_Engineered\NumberPlates\*.jpg')
print(len(path))
accuracy=100
for i in path:
    base=os.path.basename(i)
    val=os.path.splitext(base)[0]

