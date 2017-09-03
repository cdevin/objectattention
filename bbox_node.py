import numpy as np
import pickle as p
import argparse

import rospy
from  std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import random
from Featurizer import BBProposer, AlexNetFeaturizer
import time

class ImgProcessor:
    """
    Listens to a camera channel and runs a network on them. Publishes the features
    to /python_image_features.
    Currently only for tf models
    """
    def __init__(self, box_features, camera_channel='/camera_crop/image_rect_color',
                 bbox_channel='/objectattention/bbox', feat_channel='/objectattention/features',
                 similarity_channel='/objectattention/similarity'):
        """ weights should be a dictionary with variable names as keys and weights as values
        """
        print "start"
        self._camera_channel = camera_channel
        self._feature_channel = feature_channel
        self.curr_img = None
        self.fps = None
        print "session"
        self.init_model(box_features)
        self.temp = 1
        self.bbox_publisher =rospy.Publisher(bbox_channel, Float64MultiArray, queue_size =10)
        self.feat_publisher =rospy.Publisher(feat_channel, Float64MultiArray, queue_size =10)
        self.similarity_publisher =rospy.Publisher(similarity_channel, Float64MultiArray, queue_size =10)
        self.image_subscriber = rospy.Subscriber(camera_channel, Image, self._process_image)
        self.msg = None

        rospy.init_node('image_proc',anonymous=True)

    def init_model(self, box_features):
        self.proposer = BBProposer()
        self.featurizer = AlexNetFeaturizer()
        self.num_boxes = 30
        self.query = box_features

    def preprocess(self, images, draw = False):
        images = images[:,:,::-1]
        images = images.astype(np.float64)

        if np.max(images) < 2.0:
            images = images*255
        images[:,:] -= np.array([122.7717, 102.9801, 115.9465 ])
        boxes = self.proposer.extract_proposal(images)
        images[:,:] += np.array([122.7717, 102.9801, 115.9465 ])
        crops = [self.proposer.get_crop(b, images) for b in boxes]
        feats = self.featurizer.getManyFeatures(crops)
        boxes = [b for b in boxes][:self.num_boxes]
        return np.array(feats), np.array(boxes)

    def draw_boxes(self, boxes, im, c=1):
        for b in boxes:
            im = self.proposer.draw_box(b, im,c)
        return im

    def get_probs(self, feats, q):
        q = np.reshape(q, [feats.shape[1], 1])
        cos = np.abs(np.matmul(feats,q))
        exp = np.exp(cos*self.temp)
        Z = np.sum(exp)
        probs = exp/Z
        return probs,cos


    def _process_image(self, msg):
        self.msg = msg
        img = np.fromstring(msg.data, np.uint8)
        img =  np.reshape(img, (480,480,3))
        orig = img.copy()[:,:,::-1]
        feats, boxes = self.preprocess(img)
        maxbox = np.zeros((len(self.query),4))
        maxfeats = np.zeros((len(self.query),256))
        maxsim = np.zeros((len(self.query),1))
        for q in range(len(self.query)):
            probs, cos = self.get_probs(feats, self.query[q])
            nprobs = np.tile(probs, [1,4])
            softbox = np.sum(nprobs*boxes, axis = 0)
            argmax= np.argmax(probs)
            max_box = boxes[argmax]
            max_box[::2] /=img.shape[0]
            max_box[1::2] /=img.shape[1]
            maxbox[q,:] = max_box
            maxfeats[q,:] = feats[argmax]
            maxsim[q,:] = cos[argmax]
        self.prevmaxbox = max_box
        self.fps = maxbox.flatten()
        new_msg = Float64MultiArray()
        new_msg.data = self.fps
        self.bbox_publisher.publish(new_msg)
        new_msgf = Float64MultiArray()
        new_msgf.data = maxfeats.flatten()
        self.feat_publisher.publish(new_msgf)
        new_msgs = Float64MultiArray()
        new_msgs.data = maxsim.flatten()
        self.similarity_publisher.publish(new_msgs)

    def listen(self):

        while True:
            rospy.sleep(5)
            if self.msg is not None:
                print self.msg.header
                print self.fps

parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('attention')
args = parser.parse_args()
queries = np.load(args.attention)
ip = ImgProcessor(queries)
ip.listen()
