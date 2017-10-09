# objectattention

Attending to objects for robot learning.

This code provides an interface to the attention mechanism described in "Deep Object-Centric Representations forGeneralizable Robot Learning" (Devin et al. 2017) available on arxiv https://arxiv.org/abs/1708.04225. It uses a tensorflow model to attend to objects and it publishes the results to ROS at about 10-20Hz on our machines. This is intended to be used with an robot learning package such as Guided Policy Search  (https://github.com/cbfinn/gps)

This code is written for Python2 and Tensorflow 1.2, and ROS Indigo. 
For dependencies, we recommend using a python virtualenv to avoid conflicts with other pip installs. This is done in the install script:
```
./setup_linux.sh
```
Other platforms are not yet supported.


To use the example data, copy it into the data directory:
```
mkdir taskdata
mkdir taskdata/pouring
cd taskdata/pouring
wget https://people.eecs.berkeley.edu/~coline/data/pouringdata.tar.gz 
tar -xvf pouringdata.tar.gz
rm pouringdata.tar.gz
cd ../..
```
Now we will select a crop of the object to initialize the features.
```
python get_image_from_demo.py example.yaml
python scroll_box.py  taskdata/pouring/myimage.png myfeats.npy
```
The script will display an image form the demo with the RPN boxes. Click on a pixel to select the box that contains it. If multiple box contains the pixel you clicked, use the left/right arrow keys to cycle through them. Try click on the brown mug and press ENTER when your preferred box is green. This will save out the features to myfeats.npy.

If you don't want to do any finetuning, you can just use "myfeats.npy" as the attention. However, to finetune follow the following steps:
```
python process_demo_data.py example.yaml
python train_model.py example.yaml -i myfeats.npy
```
The model will save out weights periodically, to reload the network from iteration 10000 and look at it's attention, run
```
python train_model.py example.yaml -t 10000
```
This will open up an IPython notebook and you can view the soft-attended box by running
```
plt.show(plt.imshow(img))
```
To instead save the attention, run 
```
python train_model.py example.yaml -s 10000
```
which will save it in the experiment directory.

Finally, to publish the attention to ros, run:
```
python bbox_node.py taskdata/pouring/myexperiment/attention_queries.npy
```

Acknowledgements:
We thank Ronghang Hu for porting RPN from https://github.com/rbgirshick/py-faster-rcnn to tensorflow.
This work was done with the support of Huawei Technologies and the National Science Foundation.
