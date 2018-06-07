from PIL import Image
import glob
import matplotlib.pyplot as plt

# Images obtained from AT&T Laboratories Cambridge
# http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

image_list = []
dataFiles = r"C:\Users\MKC\Documents\UCSC training\Machine Learning\Team project\att_faces\*\*.pgm"

# This function changes vector data into an image plot
# v - the vector data
# show - boolean whether to show the plot or not
def vectortoimg(v,show=True):
    plt.imshow(v,interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()

# This function reads in an image, converts it into grayscale and
# decodes it into a represented matrix of integers, and stores it into a list
# dataFiles - the path of the data files
# imageList - the list to append the data to
def createImageList(dataFiles, imageList):
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
        data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
        # Append the data to the list    
        imageList.append(data)


createImageList(dataFiles, image_list)
# For drawing an image
# vectortoimg(image_list[10])
