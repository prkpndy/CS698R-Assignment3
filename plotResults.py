import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys as cs


def plotQuantity(quantityListDict, totalEpisodeCount, descriptionList):
    # This function takes in the quantityListDict and plots quantity vs episodes.
    # quantityListListDict = {envInstanceCount: quantityList}
    # quantityList is list of the quantity per episode,
    # For example it could be mean reward per episode, traintime per episode, etc.
    #
    # NOTE: len(quantityList) == totalEpisodeCount
    #
    # Since we run multiple instances of the environment, there will be variance across environments
    # so in the plot, you will plot per episode maximum, minimum and average value across all env instances
    # Basically, you need to envelop (e.g., via color) the quantity between max and min with mean value in between
    #
    # Use the descriptionList parameter to put legends, title, etc.
    # For each of the plot, create the legend on the left/right side so that it doesn't overlay on the plot lines/envelop.
    #
    # This is a generic function and can be used to plot any of the quantity of interest
    # In particular we will be using this function to plot:
    #       mean train rewards vs episodes
    #       mean evaluation rewards vs episodes
    #       total steps vs episode
    #       train time vs episode
    #       wall clock time vs episode
    #
    #
    # This function doesn't return anything

    # quantityListDict: List of dictionaries, each containing the quantity

    numPlots = len(quantityListDict)

    maxValuesList = np.zeros((totalEpisodeCount, numPlots))
    minValuesList = np.zeros((totalEpisodeCount, numPlots))
    meanValuesList = np.zeros((totalEpisodeCount, numPlots))

    for i, quantity in enumerate(quantityListDict):
        maxValuesList[:, i] = quantity.max(0)
        minValuesList[:, i] = quantity.min(0)
        meanValuesList[:, i] = quantity.mean(0)

    title = descriptionList['title']
    xLabel = descriptionList['xLabel']
    yLabel = descriptionList['yLabel']
    legend = descriptionList['legend']

    amount = 0.2
    # blue, orange, green, red, purple, dullRed
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i in range(numPlots):
        plt.plot(meanValuesList[:, i], color=colors[i], label=legend[i])
        c = cs.rgb_to_hls(*mc.to_rgb(colors[i]))
        lightColor = cs.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        plt.fill_between(range(totalEpisodeCount),
                         minValuesList[:, i], maxValuesList[:, i], color=lightColor)

    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

    return
