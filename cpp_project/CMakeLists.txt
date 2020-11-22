# https://pytorch.org/tutorials/advanced/cpp_frontend.html
cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

project(caffe2_mask_rcnn)

# we installed the pytorch via pip and python venv, then point to the pytorch folder
# you could use -DCMAKE_PREFIX_PATH=/path/to/pytorch/folder to instead
set(CMAKE_PREFIX_PATH $ENV{HOME}/workspace/unbox/unbox_cpp_caffe2/python_project/venv/lib64/python3.6/site-packages/torch)

# point to the mkl_cblas.h file
include_directories(./)

# point to the include folder of protobuf
include_directories($ENV{HOME}/.local/include)

# point to the lib folder of protobuf
link_directories($ENV{HOME}/.local/lib)

find_package(Torch REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(caffe2_mask_rcnn caffe2_mask_rcnn.cpp)
target_link_libraries(
  caffe2_mask_rcnn
  "${TORCH_LIBRARIES}" gflags glog protobuf ${OpenCV_LIBS})
set_property(TARGET caffe2_mask_rcnn PROPERTY CXX_STANDARD 14)
