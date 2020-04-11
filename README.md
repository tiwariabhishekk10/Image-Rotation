# Image-Rotation

# Problem 
The aim is to build a model to predict the appropriate angle of a clicked image.

# Methodology
To build the model:
1. Python package beautifulsoup() was used to build the scrapper for web scrapping the images from internet. 
2. Python package OpenCv() was used to rotate all the scrapped images randomly between angle 0 to 359. 
3. Finally Convolutional Neural Network (CNN) model is used for predicting the rotated angle of the image. 

# Analysis
With the help of web scrapping images were scrapped from several websites. OpenCv() was used to rotate image randomly. The input image was the rotated image and dependent variable was the rotated angle which was generated after rotation. Both the independent and dependent variables were then used in CNN for predicting the angle. Two fold cross-validations were used, dividing the data into train and test in the ratio of 75:25. For the analysis, regression was used as a output of the CNN. 
