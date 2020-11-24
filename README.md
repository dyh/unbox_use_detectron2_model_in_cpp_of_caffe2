# AI开箱 在C++中使用Detectron2的Mask R-CNN

### 欢迎订阅我的频道
- [bilibili频道](https://space.bilibili.com/326361150)
- [youtube频道](https://youtube.com/channel/UCAebg3DDFtidQJ0Jp20kyaw)

# 视频

- 为保障项目复现，本视频在虚拟机下录制，系统: ubuntu-18.04.5-desktop-amd64.iso
- 虚拟机磁盘空间，建议分配40G

bilibili


# 系统需求

- ubuntu 18.04
- python 3.6
- cuda 10.1
- cudnn 8.0.5


# 第一部分：转换为 caffe2 模型

## python 程序环境配置

1. 下载代码

    安装 git
    ```
    sudo apt install git
    ```
    下载代码
    ```
    git clone https://github.com/dyh/unbox_use_detectron2_model_in_cpp_of_caffe2.git unbox_cpp_caffe2
    ```
   
2. 进入目录
    ```
    cd unbox_cpp_caffe2/python_project/
    ```
   
3. 创建 python 虚拟环境
    ```
    sudo apt install python3-venv
    python3 -m venv venv
    ```

4. 激活虚拟环境
   ```
   source venv/bin/activate
   ```

5. 升级pip

   ```
   python -m pip install --upgrade pip
   ```

6. 安装软件包

    1. 安装gcc
        ```
        sudo apt install gcc
        ```

    2. 安装CUDA和CUDNN
    
        ```
        下载 http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
        sudo bash cuda_10.1.243_418.87.00_linux.run

        下载 https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/10.1_20201106/cudnn-10.1-linux-x64-v8.0.5.39.tgz
        解压，将include目录和lib64目录下的文件拷贝至 /usr/local/cuda 对应目录
        ```

    3. 安装 torch==1.6.0+cu101
    
        ```
        pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
        ```
    
    4. 安装 detectron2==0.3+cu101
    
        ```
        sudo apt install python3.6-dev
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
        ```
       
    5. 安装其他包

        ```
        sudo apt-get install graphviz
        pip install opencv-python==4.4.0.46
        pip install onnx==1.8.0
        pip install protobuf==3.14.0
        ```

7. 下载训练好的 Mask R-CNN weights 文件
    > model_0124999.pth，334.8 MB，这是我在上一期视频中，使用Mask R-CNN检测隧道裂缝的权重文件，包含2个分类：裂缝fissure和渗水water
    - 下载链接1: https://pan.baidu.com/s/1BqUTgciTeDxng21dYcMlag 提取码: puaq
    - 下载链接2: https://drive.google.com/file/d/1SLs-dCHibUMJY0dgcbkAb82h-Yxm6gAs/view?usp=sharing
    - 形成 ./python_project/weights/model_0124999.pth 的目录结构

8. 关于训练图片和标注文件

    > python_project/images/train 目录中已经包含训练图片和标注文件，关于 Mask R-CNN 如何标注可以参考这个视频：

    - [bilibili](https://www.bilibili.com/video/BV1DT4y1F7yG)
    - [youtube](https://www.youtube.com/watch?v=u4YpOLUxE9E)


## 运行 python 程序转换模型

```
python caffe2_converter.py
```

程序运行成功后，将在 python_project/output 目录生成 model.pb 和 model_init.pb 文件

# 第二部分：使用 C++ 调用模型


## C++ 程序环境配置

1. 进入目录

    ```
    cd ../cpp_project/
    ```

2. 安装依赖项

    ```
    sudo apt install libgflags-dev libgoogle-glog-dev libopencv-dev
    pip install mkl-include
    ```

3. 安装 protobuf

    ```
    下载 https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-cpp-3.11.4.tar.gz
    tar xf protobuf-cpp-3.11.4.tar.gz
    cd protobuf-3.11.4
    export CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
    ./configure --prefix=$HOME/.local && make && make install
    ```

4. 配置 CMakeLists.txt

    > 回到 cpp_project 目录，修改 CMakeLists.txt 文件中的内容

    1. 配置 pytorch 路径
        ```
        修改路径
        set(CMAKE_PREFIX_PATH $ENV{HOME}/workspace/unbox/unbox_cpp_caffe2/python_project/venv/lib64/python3.6/site-packages/torch) 
        指向 python 虚拟环境中的 pytorch 安装目录
        ```
    
    2. 配置 protobuf 路径
        ```
        # point to the include folder of protobuf
        include_directories($ENV{HOME}/.local/include)
        # point to the lib folder of protobuf
        link_directories($ENV{HOME}/.local/lib)
        ```

## 运行 C++ 程序检测图片

1. 编译

    > 回到 cpp_project 目录

    安装cmake
    ```
    sudo apt install cmake
    mkdir build && cd build
    cmake .. && make
    ```
   
2. 运行

    ```
    ./caffe2_mask_rcnn
    ```
   
# 官方参考

- https://github.com/facebookresearch/detectron2/tree/master/tools/deploy
- https://detectron2.readthedocs.io/tutorials/deployment.html

---

# Unbox AI, use Mask R-CNN of detectron2 in C++

### welcome to subscribe my channel
- [youtube channel](https://youtube.com/channel/UCAebg3DDFtidQJ0Jp20kyaw)
- [bilibili channel](https://space.bilibili.com/326361150)

# Video

- 为保障项目复现，本视频在虚拟机下录制，系统: ubuntu-18.04.5-desktop-amd64.iso
- 虚拟机磁盘空间，建议分配40G

youtube


# System Requirements

- ubuntu 18.04
- python 3.6
- cuda 10.1
- cudnn 8.0.5


# Chapter 1: Convert to Caffe2 Model

## Python Environment configuration

1. download source code

    install git
    ```
    sudo apt install git
    ```
    clone source code to unbox_cpp_caffe2 folder
    ```
    git clone https://github.com/dyh/unbox_use_detectron2_model_in_cpp_of_caffe2.git unbox_cpp_caffe2
    ```
   
2. enter the python project folder
    ```
    cd unbox_cpp_caffe2/python_project/
    ```
   
3. create the virtual environment
    ```
    sudo apt install python3-venv
    python3 -m venv venv
    ```

4. activate virtual environment
   ```
   source venv/bin/activate
   ```

5. upgrade pip

   ```
   python -m pip install --upgrade pip
   ```

6. install software packages

    1. install gcc
        ```
        sudo apt install gcc
        ```

    2. install CUDA and CUDNN
    
        ```
        download file: http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
        sudo bash cuda_10.1.243_418.87.00_linux.run

        download file: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/10.1_20201106/cudnn-10.1-linux-x64-v8.0.5.39.tgz
        unzip files, copy the files of include and lib64 to /usr/local/cuda/include and /usr/local/cuda/lib64 folder
        ```

    3. install torch==1.6.0+cu101
    
        ```
        pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
        ```
    
    4. install detectron2==0.3+cu101
    
        ```
        sudo apt install python3.6-dev
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
        ```
       
    5. install other software packages

        ```
        sudo apt-get install graphviz
        pip install opencv-python==4.4.0.46
        pip install onnx==1.8.0
        pip install protobuf==3.14.0
        ```

7. download pre-trained weights file of Mask R-CNN
    > model_0124999.pth, 334.8MB, in my last video, this is a weights file that uses Mask R-CNN to detect tunnel fissure, including 2 categories: fissure and water

    - download link 1: https://drive.google.com/file/d/1SLs-dCHibUMJY0dgcbkAb82h-Yxm6gAs/view?usp=sharing
    - download link 2: https://pan.baidu.com/s/1BqUTgciTeDxng21dYcMlag password: puaq
    - make the directory structure as ./python_project/weights/model_0124999.pth

8. About sample images and annotation files

    > The python_project/images/train directory already contains sample images and annotation files. You can refer to this video about Mask R-CNN annotation:

    - [youtube](https://www.youtube.com/watch?v=u4YpOLUxE9E)
    - [bilibili](https://www.bilibili.com/video/BV1DT4y1F7yG)


## Run Python program to convert the model file & weights file

```
python caffe2_converter.py
```

when the program runs successfully, model.pb and model_init.pb files are generated in the python_project/output directory

# Chapter 2: Use C++ to load weights file


## C++ Environment Configuration

1. enter the C++ project folder

    ```
    cd ../cpp_project/
    ```

2. install dependency packages

    ```
    sudo apt install libgflags-dev libgoogle-glog-dev libopencv-dev
    pip install mkl-include
    ```

3. install protobuf

    ```
    download file: https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-cpp-3.11.4.tar.gz
    tar xf protobuf-cpp-3.11.4.tar.gz
    cd protobuf-3.11.4
    export CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
    ./configure --prefix=$HOME/.local && make && make install
    ```

4. configure the CMakeLists.txt

    > return to the cpp_project directory and modify the contents of the CMakeLists.txt file

    1. configure pytorch path
        ```
        change path: 
        set(CMAKE_PREFIX_PATH $ENV{HOME}/workspace/unbox/unbox_cpp_caffe2/python_project/venv/lib64/python3.6/site-packages/torch) 
        Point to the pytorch installation directory in the python virtual environment
        ```
    
    2. configure protobuf path
        ```
        # point to the include folder of protobuf
        include_directories($ENV{HOME}/.local/include)
        # point to the lib folder of protobuf
        link_directories($ENV{HOME}/.local/lib)
        ```

## Run C++ Program to Detect Images

1. compile

    > return to the cpp_project directory
    
    install cmake
    ```
    sudo apt install cmake
    mkdir build && cd build
    cmake .. && make
    ```
   
2. run

    ```
    ./caffe2_mask_rcnn
    ```
   
# Official Reference

- https://github.com/facebookresearch/detectron2/tree/master/tools/deploy
- https://detectron2.readthedocs.io/tutorials/deployment.html