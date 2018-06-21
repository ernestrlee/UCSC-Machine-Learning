import numpy as np
from ScatterPlot import *
from readImageFiles import *
from ReadExcel import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics

dataFiles = r"/Users/sojena/training/machine-learning/project/data/*/*.pgm"
excelfile = r"/Users/sojena/training/machine-learning/project/UCSC-Machine-Learning/class_labels_for_att_faces.xlsx"
sheets = getSheetNames(excelfile)
# print(sheets)

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

# convert Y  = 1 and N = 0;
TB = np.full(T.shape,-1).astype(int);
for i,row in enumerate(T):
    for j in range(np.alen(row)):
        if T[i,j] == 'Y':
            # 1 for positive
            TB[i,j] = 1;
        else:
            # 0 for negative
            TB[i,j] = 0;

# Separate training and testing class labels
Ttrain = TB[:300]
Ttest = TB[300:]
# Prepare the training data
TtrainSmile = Ttrain[:, 0]
TtrainBlink = Ttrain[:, 1]
TtrainGood = Ttrain[:, 2]
print("Traing good:")
print(TtrainGood)
print(TtrainGood.shape)
# Prepare the testing data
TtestSmile = Ttest[:, 0]
TtestBlink = Ttest[:, 1]
# label data
# get labels - good picture - 1, bad_picture = 0
TtestGood = Ttest[:, 2]
print("Testing good:")
print(TtestGood)
print(TtestGood.shape)


# Using SKlearn to do PCA
# maximize variance
pca = PCA(n_components=100);
# dimensionality reduction
XTrain_reduced = pca.fit_transform(Xtrain);

print("PC \n",XTrain_reduced.shape);

# plot explained variance ratio - dimensions selected is close to 90% variance
with plt.style.context('bmh'):
    plt.figure(figsize=(8, 6));
    plt.title('Cumulative Explained Variance Ratio');
    plt.plot(pca.explained_variance_ratio_.cumsum());
    plt.show();

print("PCA components shape \n",pca.components_.shape)


# plot a reduced dimensions face - random 5 faces
plt.figure(figsize=(10, 4));
plt.suptitle('Reduced Dimensions Faces');
# plots 5 random faces that has been transformed across d eigen vectors
for imageIndex in range(5):
    plt.subplot(1,5, imageIndex +1)
    plt.imshow(pca.components_[np.random.randint(1,pca.components_.shape[0])].reshape(112, 92), cmap=plt.cm.gray)
    plt.grid(False);
    plt.xticks([]);
    plt.yticks([]);
# show reduced dimension faces
plt.show();

#-------------------------------------------------
# use logistic regression for classification
log_reg = LogisticRegression();

log_reg.fit(XTrain_reduced,TtrainGood);

# pca test data
pca_test = PCA(n_components=100);
XTest_reduced = pca_test.fit_transform(Xtest);

print("PCA test\n",XTest_reduced.shape);

# probability
#y_prob = log_reg.predict_proba(XTest_reduced);
#print("y_prob.shape",y_prob.shape);
#print(y_prob);

#performance
score = log_reg.score(XTest_reduced,TtestGood);
print("Score \n",score);

predictions = log_reg.predict(XTest_reduced);

index = 0
misclassifiedIndexes = []
for label, predict in zip(TtestGood, predictions):
    if label != predict:
        misclassifiedIndexes.append(index)
        index +=1

plt.figure(figsize=(20,4))
# plot first 5 misclassified images
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(pca.components_[badIndex], (112,92)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], TtestGood[badIndex]), fontsize = 10)

plt.show();

# confusion matrix

cm = metrics.confusion_matrix(TtestGood,predictions);

plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 10)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["0", "1"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 10)
plt.xlabel('Predicted label', size = 10)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x),
  horizontalalignment='center',
  verticalalignment='center')
plt.show();



#plt.figure(figsize=(12,12));
#plt.plot(XTest_reduced,y_prob[:,0],"g-",label="Smiley Face");
#plt.plot(XTest_reduced,y_prob[:,1],"r-",label="Not Smiley Face");
#plt.show();


# Draw a plot of the data
#plt.scatter(p12[:, 0], p12[:, 1])
#plt.show()
#drawScatter(p12, TtrainGood, 0, 1, "Scatter plot for classification of good picture(Green) or bad picture(Red)\n",
#            "P1", "P2")


''''
# Count total number of samples
Nn = len(np.array(TtrainGood[TtrainGood == 0]))
Np = len(np.array(TtrainGood[TtrainGood == 1]))
print("Nn:")
print(Nn)
print("Np:")
print(Np)

# Create an array of the separated principal components
Pn = np.array(p12[TtrainGood == 0])
Pp = np.array(p12[TtrainGood == 1])
print("Pn:")
# print(Pn)
print(Pn.shape)
print("Pp:")
# print(Pp)
print(Pp.shape)


# Load the positive and negative testing data
Xtestp = Xtest[TtestGood == 1]
Xtestn = Xtest[TtestGood == 0]
print(Xtestp.shape)
print(Xtestn.shape)

# Select random images from test data to display,
# one from positive class and one from negative class
xp = Xtestp[8]
xn = Xtestn[8]
# Display the images
vectortoimg(xp)
vectortoimg(xn)
'''