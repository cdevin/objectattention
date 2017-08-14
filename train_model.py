import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pickle as pic
import tensorflow as tf
import yaml
from Featurizer import BBProposer, AlexNetFeaturizer

def init_weights(shape, name=None):
    return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.01))

def init_bias(shape, name=None):
    return tf.get_variable(name, initializer=tf.zeros(shape, dtype='float'))

def get_mlp_layers(mlp_input, number_layers, dimension_hidden, name_pref=''):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name=name_pref+'w_' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name=name_pref+'b_' + str(layer_step))
        weights.append(cur_weight)
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top, weights, biases

class ProposalModel:
    def __init__(self, dim_input=7, dim_ee= 9,dim_u =7, batch_size=32, num_boxes=20, num_queries=1, im_width=810):
        self.proposer = BBProposer()
        self.featurizer = AlexNetFeaturizer()
        self.graph = tf.Graph()
        self.dim_input = dim_input
        self.dim_ee =dim_ee
        self.dim_u = dim_u
        self.batch_size = batch_size
        self.num_boxes = num_boxes
        self.im_width= im_width
        self.num_queries= num_queries
        with self.graph.as_default():
            self.init_model()
            self.init_optimizer()
            self.sess = tf.Session(graph=self.graph)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)


    def init_model(self):
        dim_hiddenu = [80,80,self.dim_u]
        n_layers = len(dim_hiddenu)
        state_input = tf.placeholder("float", [None, self.dim_input], name='state_input')
        feat_list = tf.placeholder("float", [None,self.num_boxes, self.featurizer.num_features], name='feat_input')
        boxes = tf.placeholder("float", [None,self.num_boxes, 4], name='feat_input')
        ee_output = tf.placeholder("float", [None, self.dim_ee], name='ee_output')
        u_output = tf.placeholder("float", [None, self.dim_u], name='u_output')
        feat_len = self.featurizer.num_features
        queries = []
        arg_boxes = []
        batch_num =tf.shape(feat_list)[0]
        normed_feats = feat_list
        boxes = tf.reshape(boxes, [-1, self.num_boxes,4])
        entropies = []
        probs = []
        arg_feats = []
        for i in range(self.num_queries):
            w = init_weights([self.featurizer.num_features], name='query'+str(i))
            queries.append(w)
            tiled_w = tf.tile(w, [batch_num])
            reshaped_w = tf.reshape(tiled_w, [-1, feat_len,1 ])
            print reshaped_w
            self.reshaped_w = reshaped_w
            self.tiled_w = tiled_w
            cosine_similarity = tf.abs(tf.matmul(normed_feats, reshaped_w))
            temp = 1
            exp = tf.reshape(tf.exp(cosine_similarity*temp), [-1, self.num_boxes])
            Z = tf.tile(tf.reduce_sum(exp, 1, keep_dims=True),[1,self.num_boxes])
            prob1 = tf.reshape(exp/Z, [-1, self.num_boxes, 1])
            prob = tf.tile(prob1, [1,1,4])
            arg_box = tf.reduce_sum(prob*boxes,1)
            prob = tf.tile(prob1, [1,1,256])
            arg_feat = tf.reduce_sum(prob*normed_feats,1)
            arg_feats.append(arg_feat)
            arg_boxes.append(arg_box)
            print cosine_similarity
            entropy = -tf.reduce_sum(prob*tf.log(prob))
            entropies.append(entropy)
            prob1 = tf.reshape(prob1, [-1, self.num_boxes])
            probs.append(prob1)
        fc_input = tf.concat(axis=1, values=arg_boxes+[state_input])#+probs+arg_feats)
        u_pred, weights_FC, biases_FC = get_mlp_layers(fc_input, n_layers, dim_hiddenu, name_pref='u')
        fc_vars = weights_FC + biases_FC
        self.loss_u = tf.nn.l2_loss(u_pred-u_output)
        self.true_loss = self.loss_u
        loss = self.true_loss+tf.reduce_sum(entropies)/50000
        all_vars = tf.global_variables()
        self.all_variables ={v.name: v for v in all_vars}
        self.loss = loss
        self.state_input = state_input
        self.feat_list = feat_list
        self.boxes = boxes
        self.queries = queries
        self.probs = probs
        self.ee_output = ee_output
        self.u_output = u_output
        self.arg_boxes = arg_boxes
        self.u_pred = u_pred
        self.fc_input = fc_input

    def init_optimizer(self):
        self.opt_op = tf.train.AdamOptimizer(learning_rate=0.001,
                                             beta1=0.9).minimize(self.loss, var_list=self.all_variables.values())

    def preprocess(self, images, draw = False):
        h,w,c = images.shape
        images = images[:,:,::-1]
        images = images.astype(np.float64)
        images[:,:] -= np.array([122.7717, 102.9801, 115.9465 ])
        im = images
        boxes = self.proposer.extract_proposal(im)
        images[:,:] += np.array([122.7717, 102.9801, 115.9465 ])

        crops = [self.proposer.get_crop(b, im) for b in boxes]
        feats = [self.featurizer.getFeatures(c) for c in crops]
        boxes = [b for b in boxes][:self.num_boxes]
        if draw:
            for b in boxes:
                self.proposer.draw_box(b, im, 1)
        while len(feats) < self.num_boxes:
            feats.append(np.zeros(256))
            boxes.append(np.zeros(4))

        if draw:
            return np.array(feats), np.array(boxes), im
        else:
            return np.array(feats), np.array(boxes)

    def train_model(self, X, feats, boxes,  U, vX, vfeats, vboxes,  vU, iters= 5000, batch_size=None, demo_dir='', exp_name='exp'):
        if batch_size is None:
            batch_size = self.batch_size
        N = feats.shape[0]
        batches_per_epoch = np.floor(N/batch_size)
        indices = np.arange(N)
        train_indices = indices
        np.random.shuffle(indices)
        best_val = float('inf')
        best_val_id = 0
        average_loss = 0
        with self.graph.as_default():
            for i in range(iters):
                start_idx = int(i * batch_size %
                                (batches_per_epoch * batch_size))
                idx_i = train_indices[start_idx:start_idx+batch_size]
                feed_dict = {self.state_input: X[idx_i],
                             self.feat_list: feats[idx_i],
                             self.boxes : boxes[idx_i],
                             self.u_output : U[idx_i],
                         }
                loss, _ = self.sess.run([self.loss,self.opt_op], feed_dict)
                average_loss += loss
                if i == 0 or (i+1) % 500 == 0:
                    print 'tensorflow iteration', i+1,'   average train loss',average_loss / min(i+1, 500)
                    average_loss = 0
                    feed_dict = {self.state_input: vX,
                                 self.feat_list: vfeats,
                                 self.boxes : vboxes,
                                 self.u_output : vU,
                             }
                    valloss = self.sess.run([self.loss], feed_dict)[0]
                    print "val loss", valloss
                    var_dict = {}
                    vs = tf.global_variables()
                    for v in vs:
                        var_dict[v.name] = self.sess.run(v)
                    with open(demo_dir+'/'+exp_name+'/weights_iter'+str(i+1)+'.pkl', 'wb') as f:
                        pic.dump(var_dict, f)
                    if valloss < best_val:
                        best_val = valloss
                        best_val_id = i+1
                        print "BEST___________________", i+1

    def assign_weights(self, weights_file='best_push_weights.pkl'):
        with self.graph.as_default():
            with open(weights_file, 'rb') as f:
                var_dict = pic.load( f)
            vs = tf.global_variables()
            for v in vs:
                self.sess.run(v.assign(var_dict[v.name]))


    def eval_model(self, img):
        feats, boxes, im = self.preprocess(img, draw=True)
        feed_dict = {self.feat_list: [feats],
                     self.boxes : [boxes],
                 }
        img = img[:,:,::-1]
        orig = img.copy()
        if np.max(img) < 2.0:
            img = img*255
        img = img.astype(np.uint8)
        crops = [self.proposer.get_crop(b, img.copy()) for b in boxes]
        with self.graph.as_default():
            arg_boxes = self.sess.run(self.arg_boxes, feed_dict=feed_dict)
        for b in range(boxes.shape[0]):
            self.proposer.draw_box(boxes[b], img, 2)

        probs = self.sess.run(self.probs , feed_dict)
        for b in arg_boxes:
            self.proposer.draw_box(b[0], img, 0)
        import IPython; IPython.embed()

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('paramsfile')
    parser.add_argument('-t', '--test', type=int, default=0)
    parser.add_argument('-s', '--save', type=int, default=0)
    parser.add_argument('-i', '--queryinit', type=str, default=None)

    args = parser.parse_args()
    TEST= args.test
    SAVE = args.save
    with open(args.paramsfile, 'r') as f:
        doc = yaml.load(f)

    demo = doc['taskname']
    exp = doc['experimentname']
    q = doc['hyperparams']['num_attentions']
    initq = args.queryinit
    b= doc['hyperparams']['num_boxes']
    X = np.load(demo+'/'+doc['data']['states'])
    U = np.load(demo+'/'+doc['data']['deltas'])
    model = ProposalModel(dim_input=X.shape[1],num_queries=q, num_boxes=b, dim_u = U.shape[1])
    print "Made model!"
    if SAVE:
        weights_file = demo+'/'+exp+'/weights_iter'+str(SAVE)+'.pkl'
        print weights_file
        model.assign_weights(weights_file)
        w = model.sess.run(model.queries)
        np.save(demo+'/'+exp+'/'+'attention_queries.npy', w)

    elif TEST:
        im_file = demo+'/'+doc['data']['images']
        imgs = np.load(im_file)
        imgs = imgs.reshape((-1, doc['data']['image_height'],doc['data']['image_width'], 3))
        weights_file = demo+'/'+exp+'/weights_iter'+str(TEST)+'.pkl'
        print weights_file
        model.assign_weights(weights_file)
        for t in range(100):
            img = imgs[t*20]
            model.eval_model(img)

    else:
        # Number of images to use in validation set
        val = 100
        indices = range(X.shape[0])
        np.random.shuffle(indices)
        imgs = np.load(demo+'/'+doc['data']['images'])[indices]
        feats = np.load(demo+'/'+doc['middata']['features'])[indices]
        boxes = np.load(demo+'/'+doc['middata']['boxes'])[indices]
        print "loaded data"
        X = X[indices]
        U = U[indices]
        if initq is not None:
            w = np.load(initq)
            print "Loaded W", w
            with model.graph.as_default():
                var_dict = {v.name: v for v in tf.global_variables()}
                v = var_dict['query0:0']
                model.sess.run(v.assign(w))
        np.random.seed(1309)

        if not os.path.exists(demo+'/'+exp):
            os.makedirs(demo+'/'+exp)
        model.train_model(X[val:], feats[val:], boxes[val:], U[val:], X[:val], feats[:val], boxes[:val], U[:val],iters=60000, demo_dir=demo, exp_name=exp)
