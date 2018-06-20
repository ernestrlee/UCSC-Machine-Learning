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

# TODO need to confirm this function
# This function calculates the PDF of a multinormal distribution
# X - the data set
# N - the number of samples
# sigma - the covariance matrix
# mu - the mean vector
# d - the number of dimensions (features)
def calculatePDF(X, N, sigma, mu, d):    
    e = math.e
    pi = math.pi    
    sigmaDeterminant = np.linalg.det(sigma)
    #print("det: " + str(sigmaDeterminant))
    inverseSigma = np.linalg.inv(sigma)
    #print("inv: " + str(inverseSigma))
    z = X - mu
    #print("z: " + str(z))    
    scalar1 = np.dot(z, inverseSigma)
    #print("scalar")
    scalar2 = np.dot(scalar1, z.T)
    #print("scalar: " + str(scalar2))
    rightSide = e**(-1/2 * scalar2)
    #print("right: " + str(rightSide))
    leftSide = 1 / (((2 * pi)**(d/2)) * sigmaDeterminant**(1/2))
    #print("left: " + str(leftSide))
        
    return N * (leftSide * rightSide)

# TODO need to confirm this function
# This function predicts the class label for one feature vector
# betwen two classes using a Bayesian Classifier
# (for data that is normally distributed)
# X - The individual data set to classify
# Np - Total number of positive samples
# Cp - Covariance matrix of the positive samples
# mup - Mean vector of the positive samples
# CLp - Class label for the positive
# Nn - Total number of negative samples
# Cn - Covariance matrix of the negative samples
# mun - Mean vector of the negative samples
# CLn - Class label for the negative
# returns classLabelType - Y for positive, N for negative
# returns probability
def getBayesian2ClassLabel(X, Np, Cp, mup, CLp, Nn, Cn, mun, CLn):
    # Get the number of dimension of X
    d = len(X)

    # Calculate the PDF for the positive and negative data sets
    pdfp = calculatePDF(X, Np, Cp, mup, d)
    pdfn = calculatePDF(X, Nn, Cn, mun, d)

    # Predict the class labels and compute the posterior probability
    if pdfp > pdfn:
        classLabel = CLp
        probability = pdfp / (pdfp + pdfn)
    elif pdfn > pdfp:
        classLabel = CLn
        probability = pdfn / (pdfn + pdfp)
    else:
        classLabel = "Indeterminant"
        probability = "Nan"

    return classLabel, probability

# TODO need to confirm this function
# This function computes the classification accuracy of a Bayesian
# classifier for two classes
# X - the testing data set
# T - the class labels
# V - the eigenvalues of the covariance matrix
def getBayesianAccuracy(X, T, V):
    correct = 0
    incorrect = 0
    mu = np.mean(X)

    for i in range(len(X)):
        x = X[i]
        z = x - mu

        P = np.dot(z, V.T)
        BLabel = getBayesian2ClassLabel(P, Np, Cp, mup, "Y", Nn, Cn, mun, "N")
                
        if(BLabel[0] == T[i]):
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    accuracy = correct / (correct + incorrect)
    return accuracy

# Get the image data
X = np.array(createImageList(dataFiles))
print(X)
print(X.shape)

# Separate data into training data and testing data
Xtrain = X[:300]
print("Xtrain")
print(Xtrain)
print(Xtrain.shape)
Xtest = X[300:]
print("Xtest")
print(Xtest)
print(Xtest.shape)

# Get the class data
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

# Graph components vs explained variance
pca = PCA().fit(Xtrain)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.show()

# Get the mean of the training data
mu_train = np.mean(Xtrain)
print("Mean of training data, mu_train")
print(mu_train)
print(mu_train.shape)

# Get the centered data
z_train = Xtrain - mu_train
print("Centered training data, z:")
print(z_train)
print(z_train.shape)

# Using SKlearn to do PCA with 2 components
pca = PCA(n_components=2)
pca.fit(Xtrain)

# Get the first two eigenvectors of the covariance matrix
V12 = pca.components_
print("First 2 eigenvectors, v12:")
print(V12)
print(V12.shape)

# Get the first two principal components
p12 = pca.fit_transform(Xtrain)
print("First 2 principal components, p12:")
print(p12)
print(p12.shape)

# Draw a plot of the data
plt.scatter(p12[:, 0], p12[:, 1])
plt.show()
drawScatter(p12, TtrainSmile, "N", "Y", "Scatter plot for classification of smiling picture(Green)\n or not smiling picture(Red)\n", "P1", "P2")
drawScatter(p12, TtrainBlink, "N", "Y", "Scatter plot for classification of not blinking picture(Green)\n or blinking picture(Red)\n", "P1", "P2")
drawScatter(p12, TtrainGood, "N", "Y", "Scatter plot for classification of good picture(Green)\n or bad picture(Red)\n", "P1", "P2")

# Count total number of samples for the positive and negative
# classes
Nn = len(np.array(TtrainGood[TtrainGood=="N"]))
Np = len(np.array(TtrainGood[TtrainGood=="Y"]))
print("Nn:")
print(Nn)
print("Np:")
print(Np)

# Create an array of the separated principal components for positive
# and negative classes
Pn = np.array(p12[TtrainGood=="N"])
Pp = np.array(p12[TtrainGood=="Y"])
print("Pn:")
#print(Pn)
print(Pn.shape)
print("Pp:")
#print(Pp)
print(Pp.shape)

# Calculate the mean vector for positive and negative class
mun = np.mean(Pn, axis=0)
mup = np.mean(Pp, axis=0)
print("Mu neg:")
print(mun)
print(mun.shape)
print("Mu pos:")
print(mup)
print(mup.shape)

# Calcualte the covariance matrix for positive and negative class
Cn = np.cov(Pn, rowvar=False)
Cp = np.cov(Pp, rowvar=False)
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

mu_test = np.mean(Xtest)

# Select random images from test data to display, 
# one from positive class and one from negative class
xp = Xtestp[8]
xn = Xtestn[8]
# Display the images
vectortoimg(xp)
vectortoimg(xn)

# TODO not correct needs confirmation
""" mup = np.mean(xp)
mun = np.mean(xn)

zp = xp - mup
pp = np.dot(zp, V12.T)
rp = np.dot(pp, V12)
xrecp = rp + mup

zn = xn - mun
pn = np.dot(zn, V12.T)
rn = np.dot(pn, V12)
xrecn = rn + mun """

Xrec = pca.inverse_transform(p12)

# Show image of the original and recovered images
vectortoimg(Xtrain[12])
vectortoimg(Xrec[12])


bayesianAccuracy= getBayesianAccuracy(Xtestp, TtestGood, V12)
print("Bayesian Accuracy:")
print(bayesianAccuracy)
