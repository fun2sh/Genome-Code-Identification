from NeuralNet import *
import matplotlib.pyplot as plt
import numpy as np
import cPickle, gzip

if __name__ == "__main__":

    features = []
#    nn = NeuralNet([3,1,1])
    trainfile = open("Train.txt",'r')
    for line in trainfile:
        line = line.strip().split("\t")
        features.append(np.array([[float(line[0])],[float(line[1])],[float(line[2])]]))
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
        features.append(np.array([[float(line[0])],[float(line[1])],[float(line[2])]]))
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


    etas = [0.25]
    #etas = [0.1, 0.25]
    epoch = 2000
    epoch_list = [1]
    for i in range(10,epoch+1, 10):
        epoch_list.append(i)
    accuracies_list_train = []
    accuracies_list_test = []
    #print(train[3][1])

    for e in etas:
        nn = Network([3, 10, 2])
        accuracies_train, accuracies_test = nn.SGD_train(train, epochs=epoch, eta=e, lam=0.0, verbose=True, test=test)
        accuracies_list_train.append(accuracies_train)
        accuracies_list_test.append(accuracies_test)

        print  "-" * 50


    #print accuracies_list_train, accuracies_list_test

    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    for eta, accuracies in zip (etas, accuracies_list_train):
        plt.plot(epoch_list, accuracies, label="Eta ="+ str(eta))

    plt.legend()
    plt.savefig("Train_results.png")
    plt.show()
