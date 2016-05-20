import numpy as np
import pandas as pd
import math

def spiralData(numSamples, noise):
    points = []
    n = numSamples / 2

    def genSpiral(deltaT, label):
        for i in xrange(n):
            r = i/float(n) * 5
            t = 1.75 * i / float(n) * 2 * math.pi + deltaT
            x = r * math.sin(t) + np.random.uniform(-1, 1) * noise
            y = r * math.cos(t) + np.random.uniform(-1, 1) * noise
            points.append({'x': x, 'y': y, 'label': label})

    genSpiral(0, 1)
    genSpiral(math.pi, -1)
    return pd.DataFrame(points)

def xorData(numSamples, noise):
    points = [];

    def getXORLabel(p):
        return 1 if (p['x'] * p['y'] >= 0) else -1

    for i in xrange(numSamples):
        padding = 0.3
        x, y = np.random.uniform(-5, 5, 2)
        x += padding if x > 0 else -padding
        y += padding if y > 0 else -padding
        noiseX = np.random.uniform(-5, 5) * noise
        noiseY = np.random.uniform(-5, 5) * noise
        label = getXORLabel({'x': x + noiseX, 'y': y + noiseY})
        points.append({'x': x, 'y': y, 'label': label})

    return pd.DataFrame(points)

def circleData(numSamples, noise):
    points = [];
    radius = 5;
    n = numSamples / 2

    # Returns the eucledian distance between two points in space.
    def dist(a, b):
        dx = a['x'] - b['x']
        dy = a['y'] - b['y']
        return math.sqrt(dx * dx + dy * dy)

    def getCircleLabel(p, center):
        return 1 if (dist(p, center) < (radius * 0.5)) else -1;

    def genCicle(rMin, rMax):
        for i  in xrange(n):
            r = np.random.uniform(rMin, rMax)
            angle = np.random.uniform(0, 2 * math.pi)
            x = r * math.sin(angle)
            y = r * math.cos(angle)
            noiseX = np.random.uniform(-radius, radius) * noise
            noiseY = np.random.uniform(-radius, radius) * noise
            label = getCircleLabel({'x': x + noiseX, 'y': y + noiseY}, {'x': 0, 'y': 0})
            points.append({'x': x, 'y': y, 'label': label})

    # Generate positive points inside the circle.
    genCicle(0, radius * 0.5)
    # Generate negative points outside the circle.
    genCicle(radius * 0.7, radius)

    return pd.DataFrame(points)

def gaussianData(numSamples, noise):
    points = [];
    variance = 0.5 + noise * 7
    n = numSamples / 2

    def genGauss(cx, cy, label):
        for i in xrange(n):
            x = np.random.normal(cx, variance)
            y = np.random.normal(cy, variance)
            points.append({'x': x, 'y': y, 'label': label})


    genGauss(2, 2, 1)
    genGauss(-2, -2, -1)

    return pd.DataFrame(points)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split

    data = spiralData(500, 0.1)
    train, test = train_test_split(data, test_size = 0.5)

    train.plot(kind='scatter', x='x', y='y', c='label')
    test.plot(kind='scatter', x='x', y='y', c='label')
    plt.show()


