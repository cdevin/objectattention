#!/bin/bash

#echo "Installing system dependencies..."


if source objectattention/bin/activate
then
    echo Virtualenv already exists, skipping to pip installs
else
    echo "You may be asked for your sudo password to install virtualenv."
    sudo pip install virtualenv
    virtualenv -p python3 objectattention
    sleep 2s

    if source objectattention/bin/activate
    then
	echo Succesfully created virtualenv
    else
	echo ERROR Failed to activate virtualenv
	exit 1
    fi
fi

pip3 install --ignore-installed  numpy==1.12.0
pip3 install matplotlib
pip3 install easydict
pip3 install Cython==0.26
pip3 install pyyaml
pip3 install rospy
pip3 install --upgrade --ignore-installed tensorflow-gpu==1.2.0 #pip install --upgrade tensorflow
pip3 install opencv-python
#pip3 install tkinter
pip3 install pillow
pip3 install IPython
pip3 install opencv-python
pip3 install rospkg
pip3 install catkin_pkg

echo "Downloading network weights"
wget https://people.eecs.berkeley.edu/~coline/data/bvlc_alexnet.npy .
mkdir rpn_net/model
cd rpn_net/model
wget https://people.eecs.berkeley.edu/~coline/data/fasterrcnn_vgg_coco_net.tfmodel .
cd ../util/faster_rcnn_lib && make
cd ../../..
echo "Virtual environment created! Make sure to run \`source objectattention/bin/activate\` whenever you open a new terminal and want to run programs under this package."

mkdir taskdata
mkdir taskdata/pouring
cd taskdata/pouring
wget https://people.eecs.berkeley.edu/~coline/data/pouringdata.tar.gz
tar -xvf pouringdata.tar.gz
rm pouringdata.tar.gz
cd ../..
