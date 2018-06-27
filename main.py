import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
from ScatterPlot import *
from readImageFiles import *
from ReadExcel import *
from Bayesian import *
from Histogram import *
from sklearn.decomposition import PCA

# The location of the image files
imageFiles = r"C:\Users\MKC\Documents\UCSC training\Machine Learning\Team project\att_faces\*\*.pgm"
# The location of the class labels
classLabelFile = r"C:\Users\MKC\Documents\UCSC training\Machine Learning\Team project\class_labels_for_att_faces.xlsx"
sheets=getSheetNames(classLabelFile)
#print(sheets)

# Set the positive and negative class labels
CLp = "Y"
CLn = "N"

# Get the image data
X = np.array(createImageList(imageFiles))
print(X)
print(X.shape)

# Separate data into training data and testing data sets
Xtrain = X[:300]
print("Xtrain")
print(Xtrain)
print(Xtrain.shape)
Xtest = X[300:]
print("Xtest")
print(Xtest)
print(Xtest.shape)

# Get the class data
T = readExcel(classLabelFile)
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
plt.title("Explained variance and principal components")
plt.xlabel('number of principal components')
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.show()

# Get the mean of the training data
mu_train = np.mean(Xtrain, axis = 0)
print("Mean of training data, mu_train")
print(mu_train)
print(mu_train.shape)

# Get the centered data
z_train = Xtrain - mu_train
print("Centered training data, z:")
print(z_train)
print(z_train.shape)

# Using SKlearn to do PCA with 2 components
pca = PCA(n_components = 2)
pca.fit(Xtrain)

# Get the first two eigenvectors of the covariance matrix
V12 = pca.components_
print("First 2 eigenvectors, V12:")
print(V12)
print(V12.shape)

# Get the first two principal components with sklearn
pca12 = PCA(n_components = 2)
p12 = pca12.fit_transform(Xtrain)
print("First 2 principal components, p12:")
print(p12)
print(p12.shape)

# Draw a plot of the data
# Plot for smiling or not smiling
drawScatter(p12, TtrainSmile, CLn, CLp, "Scatter plot for classification of smiling picture(Green)\n or not smiling picture(Red)\n", "P1", "P2")
# Plot for eyes open or closed
drawScatter(p12, TtrainBlink, CLn, CLp, "Scatter plot for classification of eyes open picture(Green)\n or eyes closed picture(Red)\n", "P1", "P2")
# Plot for good or bad picture
drawScatter(p12, TtrainGood, CLn, CLp, "Scatter plot for classification of good picture(Green)\n or bad picture(Red)\n", "P1", "P2")

# Load the positive and negative testing data
Xtestp = Xtest[TtestGood == CLp]
Xtestn = Xtest[TtestGood == CLn]
print("Test data shapes:")
print(Xtestp.shape)
print(Xtestn.shape)

# Get the recovered Z value with dropped components
r12 = np.dot(p12, V12)
print("r12:")
print(r12)

# Get the recovered data for X with dropped components
xrec = r12 + mu_train
print("xrec:")
print(xrec)

# Get the recovered data using sk learn as comparison
xrecsk = pca.inverse_transform(p12)

# Show image of the original and recovered images
vectortoimg(Xtrain[12], title="Original image")
vectortoimg(xrecsk[12], title="Recovered image (2 components) using SK learn")
vectortoimg(xrec[12], title="Recovered image (2 components) using manual method")


# -------- Bayesian Classifier --------

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

# Compute the Bayesian accuracy for 2 components
trainBayesAccuracy = getBayesAccuracy(Xtrain, TtrainGood, mu_train, V12, Np, Cp, mup, CLp, Nn, Cn, mun, CLn)
print("Training Bayesian Accuracy:")
print(trainBayesAccuracy)

testBayesAccuracy = getBayesAccuracy(Xtest, TtestGood, mu_train, V12, Np, Cp, mup, CLp, Nn, Cn, mun, CLn)
print("Testing Bayesian Accuracy:")
print(testBayesAccuracy)


# ---- Histogram Classifier ----

# Use Sturges rule to get number of bins
# Number of samples
Ntrain = len(Xtrain)
# Number of bins
bins = int(math.log(Ntrain, 2) + 1)
print("Sturges bins:")
print(bins)

# Get the min and max of the principal components
p1min = np.amin(p12[:, 0])
p1max = np.amax(p12[:, 0])
p2min = np.amin(p12[:, 1])
p2max = np.amax(p12[:, 1])
print("P1 min and max:")
print([p1min, p1max])
print("P2 min and max:")
print([p2min, p2max])

# Build the histograms using the training data
Hp, Hn = build2DHistogramClassifier(p12[:, 0], p12[:, 1], TtrainGood, bins, p1min, p1max, p2min, p2max, CLp, CLn)
print("Histogram data")
print("Positive class histogram:")
print(Hp)
print("Negative class histogram:")
print(Hn)

# Calculate the histogram classification accuracy of the training data
trainHistAccuracy = get2DHistAccuracy(Xtrain, TtrainGood, mu_train, V12, p1min, p1max, p2min, p2max, bins, Hp, Hn, CLp, CLn)
print("Training Histogram accuracy:")
print(trainHistAccuracy)

# Calculate histogram classification accuracy of the test data
testHistAccuracy = get2DHistAccuracy(Xtest, TtestGood, mu_train, V12, p1min, p1max, p2min, p2max, bins, Hp, Hn, CLp, CLn)
print("Testing Histogram accuracy:")
print(testHistAccuracy)
