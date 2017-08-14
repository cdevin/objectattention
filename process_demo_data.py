import numpy as np
from train_model import ProposalModel
import pickle
import yaml
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('paramsfile', metavar='f', help='the yaml file')
args = parser.parse_args()
with open(args.paramsfile, 'r') as f:
    doc = yaml.load(f)

demo_dir = doc['taskname']+'/'
num_boxes = doc['hyperparams']['num_boxes']
model = ProposalModel(num_boxes=num_boxes)
all_feat  = np.zeros((0,num_boxes,256))
all_boxes  = np.zeros((0,num_boxes,4))

all_img = np.load(demo_dir+doc['data']['images'])
all_img = np.reshape(all_img, (-1,1,doc['data']['image_height'],doc['data']['image_width'], 3))

for i in range(all_img.shape[0]):
    img = all_img[i][0]
    feats, boxes= model.preprocess(img)
    all_feat = np.concatenate((all_feat, feats.reshape([1,num_boxes,256])), axis=0)
    all_boxes = np.concatenate((all_boxes, boxes.reshape([1,num_boxes,4])), axis=0)

print "saving..."
np.save(demo_dir+doc['middata']['features'], all_feat)
np.save(demo_dir+doc['middata']['boxes'], all_boxes)
print "done!"
