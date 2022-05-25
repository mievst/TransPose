# TransPose

Source: [TransPose](https://github.com/yangsenius/TransPose)

## 1. Prepare environment
If you don't have docker, install it using the [original documentation](https://docs.docker.com/get-docker/)

### Clone repo
```
git clone https://github.com/mievst/TransPose.git
cd TransPose
```
### Pretrained models and dataset
Download and unpack pretrained models and dataset from [release](https://github.com/mievst/TransPose/releases)
You need models.zip and data.zip


## Create docker
Ubuntu:
```
docker run -it -v "$(pwd)":/TransPose ubuntu:20.04 bash
```
Windows:
```
docker run -it -v /YOUR/PATH/TO/TransPose:/TransPose ubuntu:20.04 bash
```
Note: use ubuntu:20.04 (35MB) image instead of ubuntu:latest (5GB)

## Setup docker
```
apt-get update && \
apt-get install python3 && \
apt install python-is-python3 && \
apt install python3-pip && \
apt install git && \
apt install clang && \
apt install cmake
```

## 2. Install requirements
```
cd TransPose
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. Make output (training models and files) and log (tensorboard log) directories under ${POSE_ROOT} & Make libs

   ```bash
   mkdir output log
   cd lib
   make
   ```

## 4. Run example

   ```bash
   python my_main.py
   ```

Note: To view an image, load it from a container:

'''bash
docker cp CONTAINER_ID:TransPose/YOUR_IMAGE.jpg <PATH TO DOWNLOAD FOLDER>
'''

## 5. Close docker
```
exit
docker ps -a
docker stop CONTAINER_ID
```
