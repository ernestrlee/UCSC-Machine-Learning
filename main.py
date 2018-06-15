import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
from ScatterPlot import *
from readImageFiles import *
from ReadExcel import *
from sklearn.decomposition import PCA

dataFiles = r"C:\Users\MKC\Documents\UCSC training\Machine Learning\Team project\att_faces\*\*.pgm"
excelfile=r"C:\Users\MKC\Documents\UCSC training\Machine Learning\Team project\class_labels_for_att_faces.xlsx"
sheets=getSheetNames(excelfile)
#print(sheets)

X = np.array(createImageList(dataFiles))
print(X)
print(X.shape)

# Separate training data and testing data
Xtrain = X[:300]
print("Xtrain")
print(Xtrain)
print(Xtrain.shape)
Xtest = X[300:]
print("Xtest")
print(Xtest)
print(Xtest.shape)

T = readExcel(excelfile)
print("T shape:")
print(T.shape)

# Separate training and testing class labels
Ttrain = T[:300]
Ttest = T[300:]
#Prepare the training data
TtrainSmile = Ttrain[:, 0]
TtrainBlink = Ttrain[:, 1]
TtrainGood = Ttrain[:, 2]
print("Traing good:")
print(TtrainGood)
print(TtrainGood.shape)
# Prepare the testing data
TtestSmile = Ttest[:, 0]
TtestBlink = Ttest[:, 1]
TtestGood = Ttest[:, 2]
print("Testing good:")
print(TtestGood)
print(TtestGood.shape)

# This function calculates the PDF of a multinormal distribution
# x - the data set
# numSamples - the number of samples
# sigma - the covariance matrix
# mu - the mean vector
# dimensions - the number of features compared
def calculatePDF(x, numSamples, sigma, mu, dimensions):    
    sigmaDeterminant = np.linalg.det(sigma)
    #print("det: " + str(sigmaDeterminant))
    inverseSigma = np.linalg.inv(sigma)
    #print("inv: " + str(inverseSigma))
    z = x - mu
    #print("z: " + str(z))    
    scalar1 = np.dot(z, inverseSigma)
    #print("scalar")
    scalar2 = np.dot(scalar1, z.T)
    #print("scalar: " + str(scalar2))
    rightSide = math.e**(-1/2 * scalar2)
    #print("right: " + str(rightSide))
    leftSide = 1 / (((2 * math.pi)**(dimensions/2)) * sigmaDeterminant**(1/2))
    #print("left: " + str(leftSide))
        
    return numSamples * (leftSide * rightSide)

# This function finds the class label
# classLabelType - boolean, true means positive class, false means negative class
def getBayesianLabel(X, Y):
    x = [X, Y]
    pdfp = calculatePDF(x, Np, cp, mup, 2)
    pdfn = calculatePDF(x, Nn, cn, mun, 2)

    if pdfp > pdfn:
        classLabel = "Y"
        probability = pdfp / (pdfp + pdfn)
    elif pdfn > pdfp:
        classLabel = "N"
        probability = pdfn / (pdfn + pdfp)
    else:
        classLabel = "Indeterminant"
        probability = "Nan"

    return classLabel, probability

# Using SKlearn to do PCA
pca = PCA(n_components=2)
pca.fit(Xtrain)

principalComponents = pca.fit_transform(Xtrain)
print("Principal components:")
#print(principalComponents)
print(principalComponents.shape)

p12 = principalComponents

# Draw a plot of the data
plt.scatter(p12[:, 0], p12[:, 1])
plt.show()
drawScatter(p12, TtrainGood, "N", "Y", "Scatter plot for classification of good picture(Green) or bad picture(Red)\n", "P1", "P2")

# Count total number of samples
Nn = len(np.array(TtrainGood[TtrainGood=="N"]))
Np = len(np.array(TtrainGood[TtrainGood=="Y"]))
print("Nn:")
print(Nn)
print("Np:")
print(Np)

# Create an array of the separated principal components
Pn = np.array(p12[TtrainGood=="N"])
Pp = np.array(p12[TtrainGood=="Y"])
print("Pn:")
#print(Pn)
print(Pn.shape)
print("Pp:")
#print(Pp)
print(Pp.shape)

# Caculate the mean vector
mun = np.mean(Pn, axis=0)
mup = np.mean(Pp, axis=0)
print("Mu neg:")
print(mun)
print(mun.shape)
print("Mu pos:")
print(mup)
print(mup.shape)

# TODO - need to verify this calculation for covariance
# Calcualte the covariance matrix
Cn = np.cov(Pn[0], Pn[1])
Cp = np.cov(Pp[0], Pp[1])
print("Cn:")
print(Cn)
print(Cn.shape)
print("Cp:")
print(Cp)
print(Cp.shape)

# Load the positive and negative testing data
Xtestp = Xtest[TtestGood=="Y"]
Xtestn = Xtest[TtestGood=="N"]
print(Xtestp.shape)
print(Xtestn.shape)

# Select random images from test data to display, 
# one from positive class and one from negative class
xp = Xtestp[8]
xn = Xtestn[8]
# Display the images
vectortoimg(xp)
vectortoimg(xn)