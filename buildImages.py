import os
import shutil
from ImgRotation import *
from PreProcessing import printProgressBar

# Read Images and rename with X direction
pathIn = './SampleImages/'
pathOut = './Images/'
if not(os.path.exists(pathOut)):
      os.mkdir(pathOut)

l = len(os.listdir(pathIn))
i=0
errors = 0
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
for filename in os.listdir(pathIn):
      my_dest = pathOut + os.path.splitext(filename)[0] + '_x' + os.path.splitext(filename)[1]
      my_source = pathIn + filename

      # Copy and Rename X image
      # re# read image
      imgIn = cv2.imread(my_source)
      
      # Save X image
      try:
            cv2.imwrite(my_dest,imgIn,[cv2.IMWRITE_TIFF_COMPRESSION, cv2.CV_32F])
      except:
            errors += 1
      
      # Rotate - 90⁰ Clockwise
      case = 1
      imgOut = rotate_img(imgIn, case)
      
      # Save Rotated Y image
      my_dest = pathOut + os.path.splitext(filename)[0] + '_y' + os.path.splitext(filename)[1]
      try:
            cv2.imwrite(my_dest,imgOut,[cv2.IMWRITE_TIFF_COMPRESSION, cv2.CV_32F])
      except:
            errors += 1
      i+=1
      
      printProgressBar(i, l, prefix = 'Progress:', suffix = 'Complete | Errors:{:d}'.format(errors), length = 50)