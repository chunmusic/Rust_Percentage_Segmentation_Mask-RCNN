# Rust Percentage Instance Segmentation (Mask-RCNN)

- This repository is created to calculate rust percentage with instance segmentation.

- The model was trained based on Mask-RCNN with Tensorflow2 Object Detection API.

- Mask-RCNN will detect test pieces which contains rust area and non-rust area and calculate rust percentage.

- Please go to this url to download weight file (https://drive.google.com/drive/folders/1T5eKdi7I8pXphpcSJOoptylnW1H4aJO0?usp=sharing)

- Using Mask-RCNN, it can perform up to 2-3 FPS on RTX3080

- The project was implemented in World Robot Summit 2020 (WRS) competition.

## How to use

- Please run the following command to run the demo.

```python
python inference.py
```

## Preview

![Image1](https://raw.githubusercontent.com/chunmusic/Rust_Instance_Segmentation/master/output.gif)

### Reference

https://github.com/tensorflow/models
