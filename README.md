# Good picture / bad picture recognition project using machine learning

## Summary
This project uses machine learning to determine if a set of images of people's faces contain a good picture or bad picture.  A good picture is classified as a person with thier eyes open and smiling.  A bad picture is classified as a person that is either not smiling, or has their eyes closed.

Principal component analysis is used to decrease the number of dimensions.  When classifying using a histogram classifier and a Bayesian classifier, the first two principal components were kept and the data was treated as normally distributed.

Logistic regression is also used while keeping 100 components.

## Setup
Install python 3
Install any packages as necessary such as numpy, matplotlib, and sklearn
Download the image files from the AT&T Laboratories Cambridge website or from the repository
Download the class labels excel sheet from the repository.
Change the image file path to the path/folder where you stored the images on your computer
Change the class labels file path to the path where you stored your class labels excel sheet


## Running the program
The file main.py will generate example plots of the first two principal components.  Main.py will also show reconstructed images of the first two principal components.  The file will also classify the images in the training set and testing set using a histogram and Bayesian classifier, providing an approximate accuracy for each classifier.

## Collaborators
Ernest Lee,
Abhishek Banerjee,
Somyajit Jena

## Credit
Images obtained for this project were provided by AT&T Laboratories Cambridge
http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

