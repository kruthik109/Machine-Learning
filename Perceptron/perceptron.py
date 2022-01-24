

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_wb2(data):
    w = np.random.normal(loc=0.0, scale=1.0, size=(data.shape[-1] - 1))
    w = np.zeros_like(w)
    b = 0
    count = 1000
    acc_max = 0.0
    w_max = w
    b_max = b

    while count > 0:
        for i in range(0, data.shape[0]):
            row = data[i, :]
            if (row[-1] * (np.dot(w, row[:-1]) + b) <= 0.0):
                w = w + (row[:-1] * row[-1])
                b = b + row[-1]

            # print(predict(w, b, data))
            # print('data' ,data[:, -1])
        acc = accuracy(pred=predict(w, b, data), real=data[:, -1])
        #print(acc)
        if (acc > acc_max):
            acc_max = acc
            w_max = w
            b_max = b
            i_val = count

            #print(data[i])
            #loss = _hinge_loss(row[-1], predict(w, b, np.array([data[i]]))[0])
            #print('loss', loss)
            #print(f"epochs remaining: {count}, sample: {i}, acc: {acc}", end="\n")
        # if (acc >= 0.79):
        #     break

        count = count - 1
    return  w_max, b_max, 1000 - i_val

def hinge_loss( y_pred, y_true):
    val = 0
    for i in range(len(y_pred)):
        if (y_true[i]*y_pred[i]) >= 1:
            val += 0
        else:
            val += (1- (y_true[i]*y_pred[i]))
    return (val/len(y_pred))

def predict(w, b, data):
    fin = []
    hinge_loss = 0
    for i in range(0, data.shape[0]):
        row = data[i]
        pred = np.dot(row[:-1], w) + b
        if pred > 0:
            fin.append(1)
        else:
            fin.append(-1)
    return fin

def accuracy(pred, real):
    fin = (pred == real).mean()
    return fin


def cross_validation(data):
    x = []
    y = []
    count = 0
    train_hl = 0
    hinge = []
    con = 0
    for i in [.01,.01,.01,.01,.01,.01,.01,.01,.01,.01, .10,.10,.10,.10,.10,.10,.10,.10,.10,.10, .50,.50,.50,.50,.50,.50,.50,.50,.50,.50]:

        train_sum_sq = 0
        test_01 = 0
        test_sum_sq = 0

        for j in range(1):
            #print(j)
            train = data.sample(frac=i)
            test = data[~data.index.isin(train.index)]
            #print(train)
            train = train.values
            test = test.values

            w, b , k =get_wb2(train)
            #print('w', w)
            pred = predict(w, b, test)
            pred = [-1] *len(pred)
            h = hinge_loss(pred, test[:, -1])
            #print(h)
            acc = accuracy(pred, test[:, -1])
            #print('acc', acc)
            train_hl += h
            count += 1
            con += k
            #print(count)
            if count >9:
                print("Hinge loss for", i,':',train_hl/10)
                print('Average Number of Iterations to Converge:', int(con/10))
                hinge.append(train_hl/10)
                train_hl = 0
                count = 0
    plt.scatter([.01,.1,0.5], hinge)
    plt.plot([.01,.1,0.5], hinge)
    plt.ylabel("Hinge Loss")
    plt.xlabel("Training Set Percentage")
    plt.show()

if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='CS373 Homework3 NBC')
    parser.add_argument('--trainData')
    parser.add_argument('--trainLabel')
    parser.add_argument('--testData')
    parser.add_argument('--testLabel')
    args = parser.parse_args()



    #/Users/kruthikrishnappa/PycharmProjects/pythonProject/trainFolder/titanic-train.data
    #/Users/kruthikrishnappa/PycharmProjects/pythonProject/trainFolder/titanic-train.label
    #/Users/kruthikrishnappa/PycharmProjects/pythonProject/trainFolder/titanic-test.data
    #/Users/kruthikrishnappa/PycharmProjects/pythonProject/trainFolder/titanic-test.label

    X = pd.read_csv(args.trainData, delimiter=',', index_col=None, engine='python')
    X = X[X.columns].fillna(X.mode().iloc[0])
    Y = pd.read_csv(args.trainLabel, delimiter=',', index_col=None, engine='python')
    Y = Y.replace(0, -1)
    #Y = Y.replace(1, -1)
    #print(Y)
    train = pd.concat([X,Y], axis=1)
    #train = train.to_numpy()
    #cross_validation(train)

    train = train.values

    X1 = pd.read_csv(args.testData,delimiter=',', index_col=None, engine='python')
    X1 = X1[X1.columns].fillna(X1.mode().iloc[0])
    Y1 = pd.read_csv(args.testLabel,delimiter=',', index_col=None, engine='python')
    Y1 = Y1.replace(0, -1)
    test = pd.concat([X1, Y1], axis=1)
    test = test.values

    w, b, i = get_wb2(data=train)
    pred = predict(w,b,test)

    #print(pred)
    #print(test[:,-1])
    hl = hinge_loss(pred, test[:,-1])
    acc = accuracy(pred, test[:,-1])


    print('Hinge LOSS=', round(hl, 4))
    print('Test Accuracy=' ,round(acc, 4))


