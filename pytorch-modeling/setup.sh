# conda
#curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# check if file transfer is correct
#sha256sum Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#rm Miniconda3-latest-Linux-x86_64.sh
#conda or
#python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
#pip install --user kaggle
#kaggle competitions download -c global-wheat-detection

# Install pytorch
#conda install pytorch torchvision -c pytorch

# install cocoapi
#pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
#or conda install -c conda-forge pycocotools

# Download TorchVision repo to use some files from
# references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
