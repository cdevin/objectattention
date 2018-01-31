import time
import copy
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import time
import mujoco_py
from mujoco_py.mjlib import mjlib
import sys
sys.path.append('/home/coline/objectattention')

#sys.path.append('/home/coline/visual_features/sim_push/gps/python/gps/algorithm/policy_opt/')
#from tf_model_example import get_mlp_layers
import numpy as np
import matplotlib.pyplot as plt

AGENT_MUJOCO= {
    'image_width': 120,
    'image_height': 120,
}
from Featurizer import BBProposer, AlexNetFeaturizer


class SingleObjVisRewardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # print("starting init")
        utils.EzPickle.__init__(self)

        self.last_box = np.zeros(4)
        self.gripperbox = np.zeros(4)
        self._viewer_bot = mujoco_py.MjViewer(visible=True, init_width=AGENT_MUJOCO['image_width'],
                                              init_height=AGENT_MUJOCO['image_height'])
        self._viewer_bot.start()

        self.suffix = 0
        self.proposer = BBProposer()
        self.featurizer = AlexNetFeaturizer()
        # self.query = np.load("mugfeats.npy")*10
        #self.query = np.load("/home/coline/rllab/topdown_mug.npy")*50
        self.query = np.load("/home/murtaza/Documents/objectattention2/objectattention/visreward_mug_feats.npy")*50
        self.gripper_feats = np.load("/home/murtaza/Documents/objectattention2/objectattention/visreward_gripper_feats.npy")*50

        # self.cam_pos = np.array([0.435, -0.185, -0.15, 0.75, -55., 90.])    # 7DOF camera
        self.cam_pos = np.array([0.45, -0.05, -0.323, 0.95, -90., 90.])
        self.im_w = AGENT_MUJOCO['image_width']
        self.im_h = AGENT_MUJOCO['image_height']
        self.max_boxes = 10
        self.target_pos= np.array([-0.16743428, -0.15542921,  0.0403198 ,  0.04634899])
        # print("parent class")
        mujoco_env.MujocoEnv.__init__(self, '/home/murtaza/Documents/objectattention2/objectattention/visreward_world.xml', 5)
        # mujoco_env.MujocoEnv.__init__(self, '/home/larry/dev/data-collect/examples/textured.xml', 5)
        self.init_body_pos = copy.deepcopy(self.model.body_pos)
        #
        # print("done init"
    def _step(self, a):
        #vec_1 = self.get_body_com("object")-self.get_body_com("tips_arm")
        #vec_2 = self.get_body_com("object")-self.get_body_com("goal")
        #reward_near = - np.linalg.norm(vec_1)
        #reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        #print("prior reward", 2*reward_dist+0.05*reward_ctrl)

        reward_dist = -np.linalg.norm(self.last_box- self.target_pos)
        reward_near = -np.linalg.norm(self.last_box- self.gripperbox)
        #print("now reward", reward_dist+0.05*reward_ctrl)
        #the coefficients in the following line are ad hoc
        reward = reward_dist + 0.05*reward_ctrl + 0.1*reward_near
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _get_viewer(self):
        """Override mujoco_env method to put in the
        init_width and init_height

        """
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_width=200, init_height=175)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):
        # cam_pos = np.array([0.435, -0.275, -0.15, 0.55, -50., 90.])    # 7DOF camera
        cam_pos = self.cam_pos
        self.viewer.cam.lookat[0] = cam_pos[0]
        self.viewer.cam.lookat[1] = cam_pos[1]
        self.viewer.cam.lookat[2] = cam_pos[2]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_model(self):
        # qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos = self.init_qpos

        self._viewer_bot.set_model(self.model)

        self._set_cam_position(self._viewer_bot, self.cam_pos)

        # while True:
        #     self.object = np.concatenate([self.np_random.uniform(low=-0.3, high=-0.05, size=1),
        #                              self.np_random.uniform(low=0.25, high=0.65, size=1)])
        #     self.goal = np.asarray([-0.05, 0.45])
        #     if np.linalg.norm(self.object-self.goal) > 0.17: break

        # qpos[-4:-2] = self.object
        # qpos[-2:] = self.goal
        temp = copy.deepcopy(self.init_body_pos)
        idx = 3
        angle = np.random.rand(1)*np.pi/2- np.pi/4
        offset = np.array([np.cos(angle), np.sin(angle), 0])*0.2#(np.random.rand(3)-0.5)*0.4
        offset[2] = 0
        temp[idx, :] = temp[idx, :] +offset
        self.model.body_pos = temp
        self.model.step()

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        #import IPython; IPython.embed()
        obs =  self._get_obs()
        return obs

    def _plot_attention(self, img, box, c=0,save=False):
        #
        #print(probs[argmax])
        self.proposer.draw_box((box+0.5)*120, img, c, width=2)
        #self.proposer.draw_box(softbox, img, 1)
        #import IPython;IPython.embed()
        #plt.show(plt.imshow(img))
        if save:
            filename = '/home/coline/Videos/objects/imgs/sac_itr30_{0:04d}.png'.format(self.suffix)
            self.suffix+=1
            plt.imsave(filename, img)

    def _get_attention(self, boxes, feats, img, query):
        #
        q = query.copy()
        q = np.reshape(q, [feats.shape[1], 1])
        cos = np.abs(np.matmul(feats,q))
        exp = np.exp(cos)
        Z = np.sum(exp)
        probs = exp/Z
        nprobs = np.tile(probs, [1,4])
        softbox = np.sum(nprobs*boxes, axis = 0)
        argmax= np.argmax(probs)
        # print(probs[argmax])
        # self.proposer.draw_box(boxes[argmax], img, 0)
        # self.proposer.draw_box(softbox, img, 1)
        #import IPython;IPython.embed()
        #plt.show(plt.imshow(img))
        return boxes[argmax]

    def _get_obs(self):
        self._viewer_bot.loop_once()
        img_string, width, height = self._viewer_bot.get_image()#CHANGES
        img = np.fromstring(img_string, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        #plt.imsave('env.png', img)
        boxes = np.array(self.proposer.extract_proposal(img)[:self.max_boxes])
        crops = [self.proposer.get_crop(b, img) for b in boxes]
        feats = np.array([self.featurizer.getFeatures(c) for c in crops])
        boxes = boxes/120 -0.5
        sites = self.model.data.site_xpos.flatten()
        plotimg = img.copy()
        box = self._get_attention(boxes, feats, img, self.query)
        gripperbox = self._get_attention(boxes, feats, img, self.gripper_feats)
        #import IPython; IPython.embed()
        self.last_box = box.copy()
        self.last_gripperbox = gripperbox.copy()
        # self._plot_attention(plotimg, box, c= 0)
        # self._plot_attention(plotimg, gripperbox, c =1, save=True)# np.load("feats_500.npy"))# np.load('w_attention_280.npy'))

        # x1, y1, x2,y2 = box
        # xhat = np.mean([x1,x2])/120.
        # yhat = np.mean([x1,x2])/120.
        #import IPython;IPython.embed()
        #img_data = img.flatten()
        return np.concatenate([
            self.model.data.qpos.flat[:3],
            self.model.data.qvel.flat[:3],
            sites,
            #boxes.flatten(),
            #feats.flatten()
            box.flatten(),
            # np.array([xhat,yhat])
        ])
    def _set_cam_position(self, viewer, cam_pos):

        for i in range(3):
            viewer.cam.lookat[i] = cam_pos[i]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid = -1
