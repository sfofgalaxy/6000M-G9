import os
from PIL import Image
import numpy as np
 
not_RGB = []
def gci(filepath):
# recursely iterate filepath files，including children folder
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      gci(fi_d)                  
    else:
        if not (fi.endswith('.jpg') or fi.endswith('.png')):
            continue
        img=Image.open(fi_d)
        if(img.mode!='RGB'):
            not_RGB.append(fi_d)
            img = img.convert("RGB")
            img.save(fi_d)
 
# recursely iterate ../data/
gci('../data_mask/')

print(not_RGB)
