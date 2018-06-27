import numpy as np
import math


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
    #print("determinant of sigma: " + str(sigmaDeterminant))
    inverseSigma = np.linalg.inv(sigma)
    #print("inverse of sigma: " + str(inverseSigma))
    z = x - mu
    #print("z: " + str(z))    
    leftExp = np.dot(z, inverseSigma)
    #print("left side of exponent:")
    #prit(leftExp)
    rightExp = np.dot(leftExp, z.T)
    #print("right side of exponent:")
    #print(rightExp)
    rightSide = e**(-1/2 * rightExp)
    #print("right side of equation:")
    #print(rightSide)
    leftSide = 1 / (((2 * pi)**(d/2)) * sigmaDeterminant**(1/2))
    #print("left side of equation:"")
    #print(leftSide)
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


def getBayesAccuracy(X, T, mu, V, Np, Cp, mup, CLp, Nn, Cn, mun, CLn):
    """
    This function computes the classification accuracy of a Bayesian
    classifier for a data set with two classes.

    Args:
        X : array, the data set
        T : array, the class labels
        V : array, the eigenvalues of the covariance matrix
        Np : int, total number of positive samples
        Cp : array, covariance matrix of the positive samples
        mup : array, mean vector of the positive samples
        CLp : any, class label for the positive
        Nn : int, total number of negative samples
        Cn : array, covariance matrix of the negative samples
        mun : array, mean vector of the negative samples
        CLn : any, class label for the negative

    Returns:
        accuracy : float
    """
        
    correct = 0
    incorrect = 0

    for i in range(len(X)):
        x = X[i]
        z = x - mu
        p = np.dot(z, V.T)
        BLabel = getBayesian2ClassLabel(p, Np, Cp, mup, CLp, Nn, Cn, mun, CLn)
                
        if(BLabel[0] == T[i]):
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    accuracy = correct / (correct + incorrect)
    return accuracy