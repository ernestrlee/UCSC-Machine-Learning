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

def calculatePDF(x, sigma, mu):
    """
    This function calculates the probability density of a
    multinormal distribution for a single feature vector.

    Args:
        x : array, the single data set
        sigma : array, the covariance matrix for the class
        mu : array, the mean vector for the class

    Returns:
        probability density: float
    """

    # Get the number of dimensions
    d = np.alen(mu)
    e = math.e
    pi = math.pi    
    sigmaDeterminant = np.linalg.det(sigma)
    #print("det: " + str(sigmaDeterminant))
    inverseSigma = np.linalg.inv(sigma)
    #print("inv: " + str(inverseSigma))
    z = x - mu
    #print("z: " + str(z))    
    leftExp = np.dot(z, inverseSigma)
    #print("scalar")
    rightExp = np.dot(leftExp, z.T)
    #print("scalar: " + str(scalar2))
    rightSide = e**(-1/2 * rightExp)
    #print("right: " + str(rightSide))
    leftSide = 1 / (((2 * pi)**(d/2)) * sigmaDeterminant**(1/2))
    #print("left: " + str(leftSide))
    probDensity = (leftSide * rightSide)

    return probDensity


def getBayesian2ClassLabel(x, Np, Cp, mup, CLp, Nn, Cn, mun, CLn):
    """
    This function predicts the class label for one feature vector
    betwen two classes using a Bayesian Classifier (for data that
    is normally distributed).

    Args:
        x : array, the individual data set to classify
        Np : int, total number of positive samples
        Cp : array, covariance matrix of the positive samples
        mup : array, mean vector of the positive samples
        CLp : any, class label for the positive
        Nn : int, total number of negative samples
        Cn : array, covariance matrix of the negative samples
        mun : array, mean vector of the negative samples
        CLn : any, class label for the negative
    
    Returns:
        classLabelType - CLp for positive, CLn for negative
        probability: float
    """
    
    # Calculate the PDF for the positive and negative
    probp = Np * calculatePDF(x, Cp, mup)
    probn = Nn * calculatePDF(x, Cn, mun)

    # Predict the class labels and compute the posterior probability
    if probp > probn:
        classLabel = CLp
        probability = probp / (probp + probn)
    elif probn > probp:
        classLabel = CLn
        probability = probn / (probn + probp)
    else:
        classLabel = "Indeterminant"
        probability = "Nan"

    return classLabel, probability


def getBayesAccuracy(X, T, mu, V):
    """
    This function computes the classification accuracy of a Bayesian
    classifier for a data set with two classes.

    Args:
        X : array, the data set
        T : array, the class labels
        V : array, the eigenvalues of the covariance matrix

    Returns:
        accuracy : float
    """
        
    correct = 0
    incorrect = 0

    for i in range(len(X)):
        x = X[i]
        z = x - mu
        p = np.dot(z, V.T)
        BLabel = getBayesian2ClassLabel(p, Np, Cp, mup, "Y", Nn, Cn, mun, "N")
                
        if(BLabel[0] == T[i]):
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    accuracy = correct / (correct + incorrect)
    return accuracy


def build2DHistogramClassifier(X, Y, T, B, xmin, xmax, ymin, ymax, CLp, CLn):
    """
    This function builds histogram data for a 2 class data set.
    
    Args:
        X : array, the x value
        Y : array, the y value
        T : array, the class labels
        B : int, the number of bins
        xmin : float, the minimum x value of the data set
        xmax : float, the maxiumum x value of the data set
        ymin : float, the minimum y value of the data set
        ymax : float, the maxiumum y value of the data set
        CLp : any, the postive class label
        CLn : any, the negative class label

    Returns:
        Hp : array, the positive histogram data
        Hn : array, the negative histogram data

    Raises:
        ValueError: Occurs when a class label does not match the data (CLp, CLn)
    """
    
    Hp = np.zeros((B, B)).astype('int32')
    Hn = np.zeros((B, B)).astype('int32')
    He = False

    rowIndices = (np.round(((B-1)*(X-xmin)/(xmax-xmin)))).astype('int32')
    columnIndices = (np.round(((B-1)*(Y-ymin)/(ymax-ymin)))).astype('int32')
    for i, r in enumerate(rowIndices):
        c = columnIndices[i]
        if T[i] == CLp:
            Hp[r, c] += 1
        elif T[i] == CLn:
            Hn[r, c] += 1
        else:
            raise ValueError("A class label did not match.")

    return [Hp, Hn]


def getHistogramCount(X, Y, xmin, xmax, ymin, ymax, bins, Hp, Hn):
    """
    This function finds the histogram count for each bin.
    
    Args:
        X : array, the x values
        Y : array, the y values
        xmin : float, min value of x
        xmax : float, max value of x
        ymin : float, min value of y
        ymax : float, max vaule of y
        bins : int, number of bins
        Hp : array, the positive class histogram data
        Hn : array, the negative class histogram data

    Returns:
        counts : array_int, a count of the positive and negative
    """
    
    row = (np.round(((bins-1)*(X-xmin)/(xmax-xmin)))).astype('int32')
    col = (np.round(((bins-1)*(Y-ymin)/(ymax-ymin)))).astype('int32')    
    counts = Hp[row, col], Hn[row, col]
    return counts


def getHistlabel(X, Y, xmin, xmax, ymin, ymax, bins, Hp, Hn, CLp, CLn):
    """
    # This function gets the label and probability for a histogram
    # X - the x value
    # Y - the y value
    # xmin - min value in x
    # xmax - max value in x
    # ymin - min value in y
    # ymax - max value in y
    # bins - number of bins
    # Hp - positive class histogram data
    # Hn - negative class histogram data
    # CLp - the positive class label
    # CLn - the negative class label
    """
    
    counts = getHistogramCount(X, Y, xmin, xmax, ymin, ymax, bins, Hp, Hn)
    HpCount = counts[0]
    HnCount = counts[1]

    if HpCount > HnCount:
        Hlabel = CLp
        probability = HpCount / (HpCount + HnCount)
    elif HnCount > HpCount:
        Hlabel = CLn
        probability = HnCount / (HpCount + HnCount)
    else:
        Hlabel = "Indeterminant"
        probability = "Nan"

    return Hlabel, probability


def get2DHistAccuracy(X, T, mu, V, xmin, xmax, ymin, ymax, bins, Hp, Hn, CLp, CLn):
    """
    # This function calculates the accurcay of a 2D histogram classifier
    # X - the 2D data set
    # T - the class set
    # mu - the mean vector
    # V - the eigenvectors
    # xmin - the min value of x
    # xmax - the max value of x
    # ymin - the min value of y
    # ymax - the max value of y
    # bins - the number of bins
    # Hp - the positive class histogram data
    # Hn - the negative class histogram data
    # CLp - the positive class label
    # CLn - the negative class label    
    """
    
    correct = 0
    incorrect = 0

    for i in range(len(X)):
        x = X[i]
        z = x - mu
        p = np.dot(z, V.T)
        Hlabel = getHistlabel(p[0], p[1], xmin, xmax, ymin, ymax, bins, Hp, Hn, CLp, CLn)
                
        if(Hlabel[0] == T[i]):
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
mu_train = np.mean(Xtrain, axis=0)
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
print("First 2 eigenvectors, V12:")
print(V12)
print(V12.shape)

# Get the first two principal components with sklearn
pca12 = PCA(n_components=2)
p12 = pca12.fit_transform(Xtrain)
print("First 2 principal components, p12:")
print(p12)
print(p12.shape)

r12 = np.dot(p12, V12)
print("r12:")
print(r12)

xrec = r12 + mu_train
print("xrec:")
print(xrec)

# Draw a plot of the data
#plt.scatter(p12[:, 0], p12[:, 1])
#plt.show()
drawScatter(p12, TtrainSmile, "N", "Y", "Scatter plot for classification of smiling picture(Green)\n or not smiling picture(Red)\n", "P1", "P2")
drawScatter(p12, TtrainBlink, "N", "Y", "Scatter plot for classification of not blinking picture(Green)\n or blinking picture(Red)\n", "P1", "P2")
drawScatter(p12, TtrainGood, "N", "Y", "Scatter plot for classification of good picture(Green)\n or bad picture(Red)\n", "P1", "P2")

# Load the positive and negative testing data
Xtestp = Xtest[TtestGood=="Y"]
Xtestn = Xtest[TtestGood=="N"]
print("Test data shapes:")
print(Xtestp.shape)
print(Xtestn.shape)

# Get the recovered data sk learn
Xrec = pca.inverse_transform(p12)

# Show image of the original and recovered images
vectortoimg(Xtrain[12])
vectortoimg(Xrec[12])
vectortoimg(xrec[12])


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
trainBayesAccuracy = getBayesAccuracy(Xtrain, TtrainGood, mu_train, V12)
print("Training Bayesian Accuracy:")
print(trainBayesAccuracy)

testBayesAccuracy = getBayesAccuracy(Xtest, TtestGood, mu_train, V12)
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
Hp, Hn = build2DHistogramClassifier(p12[:, 0], p12[:, 1], TtrainGood, bins, p1min, p1max, p2min, p2max, "Y", "N")
print("Histogram data")
print("Positive class histogram:")
print(Hp)
print("Negative class histogram:")
print(Hn)

# Calculate the histogram classification accuracy of the training data
trainHistAccuracy = get2DHistAccuracy(Xtrain, TtrainGood, mu_train, V12, p1min, p1max, p2min, p2max, bins, Hp, Hn, "Y", "N")
print("Training Histogram accuracy:")
print(trainHistAccuracy)

# Calculate histogram classification accuracy of the test data
testHistAccuracy = get2DHistAccuracy(Xtest, TtestGood, mu_train, V12, p1min, p1max, p2min, p2max, bins, Hp, Hn, "Y", "N")
print("Testing Histogram accuracy:")
print(testHistAccuracy)
