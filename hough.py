import numpy as np
import matplotlib.pyplot as plt

def hough(points, weights = None, discretizationRadius = 1000, discretizationAngle = 180):
    """
        Computes hough transforms of the points (in 2D)
    
        Arguments:
            points {Array n * 2} -- Datapoints to represent in Hough Space 
            weights {Array n} -- Weights to associate to each points
                (default {None} - Equal weights)
            discretizationRadius {Int} -- Discretization radius Axis
            discretizationAngle {Int} -- Discretization Angle Axis
    """   
    # Polar coordinate of all points
    r = np.sqrt(points[:,0]**2 + points[:,1]**2)
    theta = np.arctan2(points[:,1], points[:,0])

    # Hough space
    houghSpace = np.zeros((discretizationRadius, discretizationAngle))
    radiusBins = np.linspace(- np.max(r), np.max(r), discretizationRadius + 1)
    angleBins = np.linspace(-np.pi, np.pi, discretizationAngle)
    for i, phi in enumerate(angleBins):
        # For the given theta compute the radius of the line cutting the points
        radPhi = np.cos(phi - theta) * r
        hist, _ = np.histogram(radPhi, bins = radiusBins, weights = weights)
        houghSpace[:, i] += hist

    return np.flip(houghSpace, 0), angleBins * 180 / np.pi, radiusBins


def displayHough(points, weights = None):
    """
        Computes hough transforms of the points and displays it
    
        Arguments:
            points -- Datapoints to project and anlyze
            weights {Array n} -- Weights to associate to each points
    """
    houghSpace, xaxis, yaxis = hough(points, weights)

    plt.figure()
    plt.imshow(houghSpace, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], aspect='auto', cmap='gray')
    plt.ylabel("Radius")
    plt.xlabel("Angle(in deg)")
    plt.show()