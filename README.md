# End to end Object detection ensemble model

- [End to end Object detection ensemble model](#end-to-end-object-detection-ensemble-model)
  - [1.Training Object detection model](#1training-object-detection-model)
  - [2. Training of the classifier model](#2-training-of-the-classifier-model)
  - [3. Implementation](#3-implementation)
    - [Dependencies](#dependencies)
    - [Project Structure](#project-structure)
  - [4. Running the Program](#4-running-the-program)
  - [5. Sample Output](#5-sample-output)


## 1.Training Object detection model

1. The training for the object detection model is done using the given google colab notebook.
[(Colab Notebook)](https://colab.research.google.com/drive/1385gzOBLpz42vPndwaIwsElxLOa19Kj7?usp=sharing)

2. The model used for fine-tuninng is [(ssd_mobilenet_v2_320x320)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)


## 2. Training of the classifier model
1. The training for the classifier model is done using the given notebook.
[(Classifier branch)](https://github.com/Anshal55/end2end_object-detection/tree/classifier_train)
## 3. Implementation
```md
This project uses <bazel> to build and run the program.
```
Bazel is a powerful open-source build and test tool that provides fast and reliable builds across programming languages and platforms. 

<br>

<b>Description</b><br>
This C++ program performs real-time object detection and classification on video frames from a camera or RTSP stream. It utilizes helper classes for image processing, inference, bounding boxes, and drawing. The program calculates and displays frames per second (FPS) and can be exited by pressing 'q'.

<br>
<br>

### Dependencies

This project has the following dependencies:

1. **TensorFlow**
   - Repository: [TensorFlow](https://github.com/tensorflow/tensorflow)
   - Commit: 24f7ee636d62e1f8d8330357f8bbd65956dfb84d

2. **glog**
   - Repository: [google/glog](https://github.com/google/glog)
   - Tag: v0.6.0

3. **gflags**
   - Repository: [gflags/gflags](https://github.com/gflags/gflags)
   - Tag: v2.2.2

4. **absl**
   - Repository: [abseil/abseil-cpp](https://github.com/abseil/abseil-cpp)
   - Commit: 98eb410c93ad059f9bba1bf43f5bb916fc92a5ea

5. **rules_foreign_cc**
    - Repository: [bazelbuild/rules_foreign_cc](https://github.com/bazelbuild/rules_foreign_cc/archive/0.1.0.zip)
    - Version: 0.1.0

6. **linux_opencv**
    - Local repository path: /usr/local

This repo (bazel workspace) makes sure that these dependencies are properly installed or accessible before building or running the program.

### Project Structure

```
.
├── README.md
├── run_obdet.sh
├── setup_opencv.sh
├── src
│   ├── app
│   │   └── tf_infer
│   │       ├── BUILD
│   │       └── tf_infer.cc
│   └── includes
│       ├── image
│       │   ├── BUILD
│       │   ├── image_helper.cc
│       │   └── image_helper.h
│       └── object_detection
│           ├── bounding_box.cc
│           ├── bounding_box.h
│           ├── BUILD
│           ├── classifier_helper.cc
│           ├── classifier_helper.h
│           ├── models
│           ├── non_max_supression.cc
│           ├── non_max_supression.h
│           ├── object_detection_helper.cc
│           └── object_detection_helper.h
├── third_party
│   ├── BUILD
│   └── opencv_linux.BUILD
├── tree.txt
└── WORKSPACE

8 directories, 26 files

4 directories, 12 files
```

- [tf_infer.cc](src/app/tf_infer/tf_infer.cc):
This is the <b>main</b> C++ file implements object detection and classification on video frames using various helper classes and functions, processing frames from a camera or an RTSP stream and displaying the results with bounding boxes and FPS information.

- [image_helper.cc](src/includes/image/image_helper.cc): 
This C++ file defines the ImageHelper class, which provides functions for preprocessing images, drawing bounding boxes on images, and displaying frames per second (FPS) information. It includes a constructor, destructor, and three member functions for image processing and visualization.

- [bounding_box.h](src/includes/object_detection/bounding_box.h):
This file defines the data type for a bounding box.

- [non_max_supression.h](src/includes/object_detection/non_max_supression.h):
This C++ header file defines functions for calculating the Intersection over Union (IoU) between bounding boxes and performing Non-Maximum Suppression (NMS) on a vector of bounding boxes. It includes implementations for IoU calculation and NMS selection based on a given IoU threshold.

- [object_detection_helper.cc](src/includes/object_detection/object_detection_helper.cc):
This C++ source file defines the implementation of an object detection helper class. It loads an object detection model, builds an interpreter, performs inference on input images, and gathers bounding boxes from the model's output based on a confidence threshold.

- [classifier_helper.cc](src/includes/object_detection/classifier_helper.cc):
This C++ source file defines the implementation of a classifier helper class. It loads a classifier model, builds an interpreter, updates the input tensor with a cropped and resized image region, performs inference, and returns true if the classification score is above a specified threshold, and false otherwise.

## 4. Running the Program
To run this program, a docker image with all the dependencies installed and a ready to run envioronment is provided.

1. Clone the repository
```sh
gh repo clone Anshal55/end2end_object-detection
```

or

```sh
git clone https://github.com/Anshal55/end2end_object-detection.git
```

2. Get and run the docker image (create container)
```sh
docker run -it -e DISPLAY --name <container-name> --device=/dev/video0:/dev/video0 --net=host --privileged -v /home/<user-name>:/home/<user-name> -v /tmp/.X11-unix:/tmp/.X11-unix -w <path/to/the/cloned/repo> anshal888/bazel_objdet:180623
```
This will run the docker container and you should be in cloned repo's main directory.
If you want to start the container again, do:
```sh
docker start -i <container-name>
```

3. Start running the program.
- To run the program first give permission to the executabel script.
```sh
chmod +x ./run_obdet.sh
```

- To run the program with webcam.
```
# to toggle the logs pass 1 or 0
./run_obdet.sh 1
```

- To run the program with rtsp stream
```
./run_obdet.sh 1 rtsp rtsp://ip:port/gateway
```
<br>
<br>

## 5. Sample Output
[Sample Video](https://youtu.be/X0AiXC9XlDo)



