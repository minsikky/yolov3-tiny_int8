# Reference Software for YOLOv3-tiny/YOLOv4-tiny
  💡This code enables you to perform int8 quantization for YOLOv3-tiny/YOLOv4-tiny and run inference with the quantized model.\
  🕯The codes are largely based on yolo2_light(https://github.com/AlexeyAB/yolo2_light/) with slight modifications.\
  ☁️A test dataset consisting of 229 images is also included (`bin/dataset`).

* How to run?
  1. Run `make_cur_list.py` in `dataset` directory. This generates `target.txt` that consists of absolute paths of the images included in the dataset.
  2. `make` compiles the executable `darknet`.
  - To evaluate the original floating point model on testset, run `yolov3-tiny-aix2022.sh` or `yolov4-tiny-aix2023.sh`.
  - To evaluate the quantized model, run `yolov3-tiny-aix2022-int8.sh` or `yolov4-tiny-aix2023.sh`.
  - To run inference on individual image, run `yolov[3/4]-tiny-aix202[2/3][-int8]-test.sh`.

* Tip
  - To save the quantized model, use flag `-save_params` at the end of command (see `yolov3-tiny-aix2022-int8.sh` and `yolov4-tiny-aix2023.sh`)
