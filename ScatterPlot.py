import numpy as np
import matplotlib.pyplot as plt

# This function draws a Scatter Plot
# P - The 2D data to plot as a 2D array
# T - The class labels (used for length)
# labeln - The negative class label
# labelp - The positive class label
# title - The title of the plot
# xlabel - The label for the x axis
# ylabel - The label for the y axis
def drawScatter(P, T, labeln, labelp, title="", xlabel="", ylabel=""):
            
    # For best effect, points should not be drawn in sequence but in random order
    np.random.seed(0)
    randomorder=np.random.permutation(np.arange(len(T)))
    randomorder=np.arange(len(T))

    # Set colors
    cols=np.zeros((len(T),4))     # Initialize matrix to hold colors
    cols[T==labeln]=[1,0,0,0.25] # Negative points are red (with opacity 0.25)
    cols[T==labelp]=[0,1,0,0.25] # Positive points are green (with opacity 0.25)

    # Draw scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.scatter(P[randomorder,1],P[randomorder,0],s=5,linewidths=0,facecolors=cols[randomorder,:],marker="o")
    ax.set_aspect('equal')

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
