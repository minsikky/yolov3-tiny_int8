# Reference Software for YOLOv3-tiny/YOLOv4-tiny

* How to run?
  - First, run `make_cur_list.py` in `dataset` directory.
  - To evaluate the original floating point model on testset, run `yolov3-tiny-aix2022.sh` or `yolov4-tiny-aix2023.sh`.
  - To evaluate the quantized model, run `yolov3-tiny-aix2022-int8.sh` or `yolov4-tiny-aix2023.sh`.
  - To run inference on individual image, run `yolov[3/4]-tiny-aix202[2/3][-int8]-test.sh`.

* Tip
  - To save the quantized model, use flag `-save_params` at the end of command (see `yolov3-tiny-aix2022-int8.sh` and yolov4-tiny-aix2023.sh`)
