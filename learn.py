import numpy as np
from sklearn.neural_network import MLPClassifier

def learn_nn(X, y):
    net = MLPClassifier(hidden_layer_sizes=(82,2), alpha=1e-5)
    net.fit(X, y)
    return net

if __name__ == '__main__':
    print('loading features from features.txt...', end='')
    features = np.loadtxt('features.txt')
    print('done')
    print('training NN...', end='')
    nn = learn_nn(features[:, :-1], features[:, -1])
    print('done')
    print(nn)