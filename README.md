# Image Colorization

This project aims to provide a professional solution for colorizing black and white images using Python. It utilizes a deep neural network implemented with OpenCV to predict the color channels of grayscale images, resulting in vibrant and realistic colorizations.

## Introduction
Image colorization is the process of adding color to black and white or grayscale images. It not only brings old photos to life but also enables artists, designers, and researchers to explore creative possibilities. This project offers an automated solution that leverages deep learning techniques to generate accurate and visually appealing colorizations.

## Functionality
The main functionality of this project is to convert black and white images to color images. It achieves this by utilizing a pre-trained Caffe model along with a prototxt file and a NumPy file. The project extracts the L channel from the grayscale image, predicts the a and b color channels, and combines them with the L channel to create a full-color image.

## Usage
To use this project, follow these steps:
1. Ensure that all the prerequisites are installed (see below).
2. Clone the repository or download the necessary files.
3. Run the `imageColorization.ipynb` Jupyter Notebook or execute the `imageColorizationGUI.py` script for an interactive GUI experience.
4. Provide the black and white image as input.
5. The project will generate a colorized version of the image and display it.

## Features
- Black and white image colorization
- Support for various image formats
- Graphical user interface for interactive usage
- Automatic prediction of color channels using a deep neural network
- Integration of the Lab color space for accurate colorization

## How to Download models
- `colorization_deploy_v2.prototxt` can be found on my repository.
- `colorization_release_v2.caffemodel` you can download the model file from [Google Drive Link](https://drive.google.com/file/d/14YmdCfcMOgfJEBNJEl6Xj1SB-RccgJBO/view?usp=sharing).

## Prerequisites
Ensure that the following dependencies are installed:
- Python 3.10 and ^
- Jupyter Notebook (for running the provided notebook)
- OpenCV
- NumPy
- Matplotlib
- Tkinter (for running the GUI)

## Technologies Used
- Python
- OpenCV (Open Source Computer Vision Library)
- Caffe (Convolutional Architecture for Fast Feature Embedding)
- NumPy (Numerical Python)
- Matplotlib (Data Visualization Library)
- Tkinter (Python GUI toolkit)

## How to Run
To run this project, follow these steps:
1. Install the prerequisites mentioned above.
2. Clone the repository or download the necessary files.
3. Open the `imageColorization.ipynb` Jupyter Notebook in your preferred environment and execute the code cells. Alternatively, run the `imageColorizationGUI.py` script for an interactive GUI experience.
4. Provide the path to the black and white image when prompted.
5. The project will generate a colorized version of the image and display it.
6. Follow the on-screen instructions in the GUI (if using) to upload and colorize images.

## Screenshot

![sdfg](https://github.com/viv3k19/imageColorization-using-Python-OpenCV/assets/82309435/79659bb5-0249-4da0-99ab-4cec08d7c284)

# Project Creator
* Vivek Malam - Feel free to contact me at viv3k.19@gmail.com for any questions or feedback.
