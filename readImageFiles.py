from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt

# This function changes vector data into an image plot
# v - the vector data
# show - boolean whether to show the plot or not
def vectortoimg(v,show=True):
    plt.imshow(v.reshape(112, 92),interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()

# Plot 100 images from the data
# X - 100 samples of the data set (i.e. [0:100])
def draw100VectorImages(X):
    print("Checking multiple training vectors by plotting images.\nBe patient:")
    plt.close('all')
    fig = plt.figure()
    nrows=10
    ncols=10
    currentImage=0
    for row in range(nrows):
        for col in range(ncols):
            plt.subplot(nrows, ncols, row*ncols+col + 1)
            vectortoimg(X[currentImage],show=False)
            currentImage += 1
    plt.show()

# This function reads in an image, converts it into grayscale and
# decodes it into a represented array of integers, and stores each image into a list
# dataFiles - the path of the data files
def createImageList(dataFiles):
    imageList = []
    for filename in glob.glob(dataFiles):
        img = Image.open(filename).convert("L")
        WIDTH, HEIGHT = img.size

        # Image size
        # print(Image size: )
        # print(WIDTH)
        # print(HEIGHT)

        # Convert image data to a list of integers
        data = list(img.getdata()) 
        # Convert that to 2D list (list of lists of integers)
        #data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
        
        # Append the data to the list    
        imageList.append(data)
    
    return imageList

# ------- For debugging -------
# Images obtained from AT&T Laboratories Cambridge
# http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
# dataFiles = r"C:\Users\MKC\Documents\UCSC training\Machine Learning\Team project\att_faces\*\*.pgm"
# X = np.array(createImageList(dataFiles))
# print(X.shape)
# For drawing an individual image
# vectortoimg(X[10])
# For drawing 100 images
# draw100VectorImages(X[300:])
