#!/bin/bash

#echo "Installing system dependencies..."


if source objectattention/bin/activate
then
    echo Virtualenv already exists, skipping to pip installs
else
    echo "You may be asked for your sudo password to install virtualenv."
    sudo pip install virtualenv
    virtualenv -p python2.7 objectattention
    sleep 2s

    if source objectattention/bin/activate
    then
	echo Succesfully created virtualenv
    else
	echo ERROR Failed to activate virtualenv
	exit 1
    fi
fi

pip install numpy==1.11.0
pip install matplotlib
pip install easydict
pip install Cython==0.26
pip install pyyaml
pip install rospy
pip install --upgrade tensorflow-gpu #pip install --upgrade tensorflow
pip install opencv-python
pip install tkinter
pip install pillow
pip install IPython
pip intall opencv-python

echo "Downloading network weights"
wget https://people.eecs.berkeley.edu/~coline/data/bvlc_alexnet.npy .
mkdir rpn_net/model
cd rpn_net/model
wget https://people.eecs.berkeley.edu/~coline/data/fasterrcnn_vgg_coco_net.tfmodel .
cd ../util/faster_rcnn_lib && make
cd ../../..
echo "Virtual environment created! Make sure to run \`source objectattention/bin/activate\` whenever you open a new terminal and want to run programs under this package."
