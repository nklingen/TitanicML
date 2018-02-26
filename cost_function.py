import numpy as np

# Create data
x = np.array([3, 2, 4, 0])
y = np.array([4, 1, 3, 1])


def cost_function(x, y, par1, par2, m):
    """
    Computes the cost function of the squared errors in linear regression

    INPUT:
        x: numpy array, data on the x-axis
        y: numpy array, data on the y-axis
        par1: float, first parameter, intersection with y
        par2: float, second parameter, slope of line
        m = int, size of dataset

    OUTPUT:
        float, The value of the cost function
    """

    # Compute cost function by vector computation
    return sum((((par1 + par2 * x) - y)**2) / (2 * m))
