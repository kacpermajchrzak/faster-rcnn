# FasterRCNN implementation

## This implementation is based on this articles:

- [Yinghan Xu. Faster R-CNN (object detection) implemented by Keras](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a)
- [Neeraj Krishna. Understanding and Implementing Faster R-CNN](https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0)

## A few words

This implementation may help you understand how Faster RCNN work. Two main files are `utils.py` and `model.py`. The `ipynb` files are for showing results of the implementation. If you don't understand the basics, please first read the articles this implementation is based on.

## How to run

1. `pip install -r requirements.txt`
2. Later go through `DataPreprocessing.ipynb` and get data which are needed for training.
3. To speed up training resize data in `resizing.ipynb`
4. If you done all of that go to `DataPreprocessing.ipynb`

## Tips

- This version is optimize for working with cuda so make sure to install torch for cuda.
- If you have a memory problem, change **BATCH_SIZE**
- To see more results change **NUMBER_OF_SAMPLES**
- Make sure to resize images to get better performance
