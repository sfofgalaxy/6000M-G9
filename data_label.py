'''
{
      "labels": [
          ["00000/img00000000.png",6],
          ["00000/img00000001.png",9],
          ... repeated for every image in the datase
          ["00049/img00049999.png",1]
      ]
}
'''

import os 
import json

labels = []
def label(filepath):
# recursely iterate filepath files，including children folder
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      label(fi_d)                  
    else:
      save = fi_d.replace('\\', '/')
      if not (fi.endswith('.jpg') or fi.endswith('.png')):
            continue
      if fi.endswith('Mask.jpg') or fi.endswith('Mask.png') or fi.startswith('mask'):
          labels.append([save, 1])
      else:
          labels.append([save, 0])
 
# recursely iterate ../data/
label('./')

# dumps data to string
info_json = json.dumps({"labels": labels},sort_keys=False, indent=4, separators=(',', ': '))
# write into file
f = open('dataset.json', 'w')
f.write(info_json)
