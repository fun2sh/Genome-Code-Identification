import numpy as np

class Network:
    def __init__(self, sizes):
        self.L = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(n, 1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n, m) for (m, n) in zip(self.sizes[:-1], self.sizes[1:])]

    def g(self, z):

        return sigmoid(z)

    def g_prime(self, z):

        return sigmoid_prime(z)

    def forward_prop(self, a):

        for (W, b) in zip(self.weights, self.biases):
            a = self.g(np.dot(W, a) + b)
        return a

    def gradC(self, a, y):
        return (a - y)

    def SGD_train(self, train, epochs, eta, lam=0.0, verbose=True, test=None):

        n_train = len(train)
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for kk in range(n_train):
                xk = train[perm[kk]][0]
                yk = train[perm[kk]][1]
                #print(xk)
                #print(yk)
                dWs, dbs = self.back_prop(xk, yk)
                self.weights = [W - eta * (dW + W * lam) for (W, dW) in zip(self.weights, dWs)]
                self.biases = [b - eta * db for (b, db) in zip(self.biases, dbs)]
            if verbose:
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    acc_train = self.evaluate(train)
                    if test is not None:
                        acc_test = self.evaluate(test)
                        print "Iteration {:4d}: Train {:10.5f}, Test {:10.5f}".format(epoch + 1, acc_train, acc_test)
                    else:
                        print "Iteration {:4d}: Train {:10.5f}".format(epoch + 1, acc_train)

    def back_prop(self, x, y):

        db_list = [np.zeros(b.shape) for b in self.biases]
        dW_list = [np.zeros(W.shape) for W in self.weights]
        delta = [None for i in self.weights + [None]]

        a = x
        a_list = [a]
        z_list = [np.zeros(a.shape)]
        #print(W.shape)
        #print(a.shape)
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            z_list.append(z)
            a = self.g(z)
            a_list.append(a)

        delta[-1] = (self.gradC(a, y)) * (self.g_prime(z_list[-1]))

        for ell in range(self.L - 2, -1, -1):

            dW_list[ell] = np.dot(delta[ell + 1], np.transpose(a_list[ell]))

            db_list[ell] = delta[ell + 1]

            rhs = (self.g_prime(z_list[ell]))
            lhs = np.dot(np.transpose(self.weights[ell]), delta[ell + 1])
            delta[ell] = lhs * rhs

        return (dW_list, db_list)

    def evaluate(self, test):
        ctr = 0
        for x, y in test:
            yhat = self.forward_prop(x)
            ctr += np.argmax(yhat) == np.argmax(y)
        return float(ctr) / float(len(test))

    def compute_cost(self, x, y):
        a = self.forward_prop(x)
        return 0.5 * np.linalg.norm(a - y) ** 2

    def gradient_checking(self, train, EPS=0.0001):
        kk = np.random.randint(0, len(train))
        xk = train[kk][0]
        yk = train[kk][1]
        dWs, dbs = self.back_prop(xk, yk)

        for ell in range(self.L - 1):
            for ii in range(self.weights[ell].shape[0]):
                for jj in range(self.weights[ell].shape[1]):
                    true_dW = dWs[ell][ii, jj]
                    foo = self.weights[ell][ii, jj]
                    self.weights[ell][ii, jj] = foo + EPS
                    num_dW = self.compute_cost(xk, yk)
                    self.weights[ell][ii, jj] = foo - EPS
                    num_dW -= self.compute_cost(xk, yk)
                    self.weights[ell][ii, jj] = foo
                    num_dW /= (2 * EPS)
                    rel_dW = np.abs(true_dW - num_dW) / np.abs(true_dW)
                    print "w: {: 12.10e}  {: 12.10e} {: 12.10e}".format(true_dW, num_dW, rel_dW)
                    rel_errors.append(rel_dW)

                true_db = dbs[ell][ii, 0]
                bar = self.biases[ell][ii, 0]
                self.biases[ell][ii, 0] = bar + EPS
                num_db = self.compute_cost(xk, yk)
                self.biases[ell][ii, 0] = bar - EPS
                num_db -= self.compute_cost(xk, yk)
                self.biases[ell][ii, 0] = bar
                num_db /= (2 * EPS)
                rel_db = np.abs(true_db - num_db) / np.abs(true_db)
                print "b: {: 12.10e}  {: 12.10e} {: 12.10e}".format(true_db, num_db, rel_db)
                rel_errors.append(rel_db)

        return rel_errors


def sigmoid(z, threshold=20):
    z = np.clip(z, -threshold, threshold)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def mnist_digit_show(flatimage, outname=None):
    import matplotlib.pyplot as plt

    image = np.reshape(flatimage, (-1, 14))

    plt.matshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    if outname:
        plt.savefig(outname)
    else:
        plt.show()

if __name__ == "__main__":

    features = []
#    nn = NeuralNet([3,1,1])
    trainfile = open("Train.txt",'r')
    for line in trainfile:
        line = line.strip().split("\t")
        features.append(np.array([[float(line[0])]]))#,[float(line[1])]]))#,[float(line[2])]]))
    trainfile.close()
    x_train = np.array(features)

    labels = []
    trainy = open("Labels.txt",'r')
    for line in trainy:
        line = line.strip().split("\t")
        labels.append(np.array([[float(line[0])],[float(line[1])]]))
    trainy.close()

    train = []
    for g in range(len(labels)):
        train.append([features[g],labels[g]])


    features = []
    testfile = open("Test.txt",'r')
    for line in testfile:
        line = line.strip().split("\t")
        features.append(np.array([[float(line[0])]]))#,[float(line[1])]]))#,[float(line[2])]]))
    testfile.close()

    labels = []
    trainy = open("TestLabels.txt",'r')
    for line in trainy:
        line = line.strip().split("\t")
        labels.append(np.array([[float(line[0])], [float(line[1])]]))
    trainy.close()

    test = []
    for g in range(len(labels)):
        test.append([features[g],labels[g]])

    #print(train[3][1])
    nn = Network([1, 10, 2])
    nn.SGD_train(train, epochs=200, eta=0.25, lam=0.0, verbose=True, test=test)
