##############
# Name: Kruthi Krishnappa
# email: krish109@purdue.edu
# Date: 10/12/2020


import pandas as pd

def get_prob(data):
    prob = {}
    zero = len(data[data['survived'] == 0])
    one = len(data[data['survived'] == 1])

    d = (data[data['Sex'] ==1])
    print(d)
    print(len(d[d['survived'] == 0]))
    print(len(d[d['survived'] == 1]))

    for i in range(len(data.columns)-1):
        avg = data.iloc[:,i].mean()

        less_zero = (data[data.iloc[:, i] <= avg])
        less_zero = less_zero[less_zero['survived'] == 0]

        greater_zero = (data[data.iloc[:, i] > avg])
        greater_zero = greater_zero[greater_zero['survived'] == 0]

        less_one = (data[data.iloc[:, i] <= avg])
        less_one = less_one[less_one['survived']==1]

        greater_one = (data[data.iloc[:, i] > avg])
        greater_one = greater_one[greater_one['survived']==1]

        lis = [avg, (len(less_zero)+1)/((zero)+2), (len(less_one)+1)/((one)+2), (len(greater_zero)+1)/((zero)+2), (len(greater_one)+1)/((one)+2)]
        prob[data.columns[i]] = lis

    return (prob)

def predict(probabilities, data):
    cols = data.columns
    pred_vals = []
    sq_loss = 0
    for i, row in data.iterrows():
        zero_prob = 1
        one_prob = 1
        for j in range(len(cols)):
            avg = probabilities[cols[j]][0]
            if row[j] > avg:
                zero_prob *= probabilities[cols[j]][3]
                one_prob *= probabilities[cols[j]][4]
            else:
                zero_prob *= probabilities[cols[j]][1]
                one_prob *= probabilities[cols[j]][2]

        if zero_prob > one_prob:
            pred_vals.append(0)
            sq_loss += (1-zero_prob) **2
        else:
            pred_vals.append(1)
            sq_loss += (1 - one_prob) ** 2

        sq_loss = sq_loss/len(data)
    return pred_vals, sq_loss



def accuracy(y_pred, y_true):
    acc = (y_pred == y_true).mean()
    return acc

def cross_validation(data):
    x = []
    y = []
    for i in [.01, .10, .50]:
        train_01 = 0
        train_sum_sq = 0
        test_01 = 0
        test_sum_sq = 0
        for j in range(10):
            train = data.sample(frac=i)
            test = data[~data.index.isin(train.index)]
            prob = get_prob(train)
            pred_train, sq_loss_train = predict(prob, train.iloc[:,:-1])
            train_acc = accuracy(pred_train, train.iloc[:,-1])
            #print('i', i, 'train_acc', train_acc)
            train_01 += (1-train_acc)
            train_sum_sq += sq_loss_train
            pred_test, sq_loss_test = predict(prob, test.iloc[:,:-1])
            test_acc = accuracy(pred_test, test.iloc[:,-1])
            #print( 'test_acc', train_acc)
            test_01 += (1 - test_acc)
            test_sum_sq += sq_loss_test

        print('training set size:',i)
        #print('train zero-one-loss', train_01/10)
        #print('train sum-squared error',train_sum_sq/10)
        #print('Zero-one-loss', test_01/10)
        print('Sum-squared error', test_sum_sq/10)
        x.append(i)
        y.append(test_sum_sq/10)


if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='CS373 Homework3 NBC')
    parser.add_argument('--trainData')
    parser.add_argument('--trainLabel')
    parser.add_argument('--testData')
    parser.add_argument('--testLabel')
    args = parser.parse_args()



    import matplotlib.pyplot as plt
    #/Users/kruthikrishnappa/PycharmProjects/pythonProject/trainFolder/titanic-train.data
    #/Users/kruthikrishnappa/PycharmProjects/pythonProject/trainFolder/titanic-train.label
    #/Users/kruthikrishnappa/PycharmProjects/pythonProject/trainFolder/titanic-test.data
    #/Users/kruthikrishnappa/PycharmProjects/pythonProject/trainFolder/titanic-test.label

    X = pd.read_csv(args.trainData, delimiter=',', index_col=None, engine='python')
    X = X[X.columns].fillna(X.mode().iloc[0])
    Y = pd.read_csv(args.trainLabel, delimiter=',', index_col=None, engine='python')
    train = pd.concat([X,Y], axis=1)

    X1 = pd.read_csv(args.testData,delimiter=',', index_col=None, engine='python')
    X1 = X1[X1.columns].fillna(X1.mode().iloc[0])
    Y1 = pd.read_csv(args.testLabel,delimiter=',', index_col=None, engine='python')
    test = pd.concat([X1, Y1], axis=1)

    probs = get_prob(train)
    print(probs)
    pred, sq_loss = predict(probs, X1)
    acc = accuracy(pred, Y1['survived'])


    print('ZERO-ONE LOSS=', round(1-acc, 4))
    print('SQUARED LOSS=', round(sq_loss, 4))
    print('Test Accuracy=', round(acc, 4))








