# Machine Learning Project
## Intro
This project was undertaken as a final project for a class on machine learning at Clarkson University for the fall 24 semester. The goal of this project was to conduct classification of audio samples into 10 different classes. This model is meant to be lightweight with the hopes of potential implementation on a microcontroller. It uses a mix of the mean RMS amplitude and MFCCs as a feature set.

## Usage
Start by running **split_files.py** which will create two folders containing the train and test set of samples from the original Free Spoken Digit Dataset. It will also create two csv files containing the label and file name information for both of the datasets. Then run **Audio_CLassification_org.ipynb** which will load the datasets and run a parameter search on the number of MFCCs vs Model F1 Score.

## Acknowledgments
This project was conducted with aid from Dr. Imtiaz as well as the help of online resources including ChatGPT. The dataset used for this project was the **Free Spoken Digit Dataset** which can be found here

https://github.com/Jakobovski/free-spoken-digit-dataset

The signal processing code is a modified form of this

https://github.com/ksrvap/Audio-classification-using-SVM-and-CNN

project which proved an invalubale resource on conducting audio processing within python.
