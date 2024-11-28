#!/bin/bash
pip install --upgrade pip
python -m venv env
source env/bin/activate
pip install tensorflow > /dev/null 2>&1
echo 'Tensorflow Installed'
pip install keras > /dev/null 2>&1
echo 'Keras Installed'
pip install praw > /dev/null 2>&1
echo 'Praw Installed'
pip install scikit-learn > /dev/null 2>&1
echo 'scikit-learn Installed'
pip install Pillow > /dev/null 2>&1
echo 'Pillow Installed'
pip install pickle > /dev/null 2>&1
echo 'Pickle Installed'
pip install python-dotenv > /dev/null 2>&1
echo 'dotenv Installed'
pip install pandas > /dev/null 2>&1
echo 'Pandas Installed'
cd ~/project
mkdir -p ./images
sudo chmod -R +rw ./images
sudo chown -R $USER:$USER ./images/
