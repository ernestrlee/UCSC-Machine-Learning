import numpy as np


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