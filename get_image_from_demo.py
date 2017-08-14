import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('paramsfile', metavar='f', help='the yaml file')
args = parser.parse_args()
with open(args.paramsfile, 'r') as f:
    doc = yaml.load(f)

demo_dir = doc['taskname']+'/'
all_img = np.load(demo_dir+doc['data']['images'])
all_img = np.reshape(all_img, (-1,doc['data']['image_height'],doc['data']['image_width'], 3)).astype(np.uint8)
plt.imsave(demo_dir+'/myimage.png', all_img[0][:,:,::-1])
