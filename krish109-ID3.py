##############
# Name: Kruthi Krishnappa
# email: krish109@purdue.edu
# Date: 10/12/2020

import numpy as np
import pandas as pd
import sys
import os
import math

#calculate entropy
def entropy(freqs):
    all_freq = sum(freqs)
    entropy = 0
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy

#calculate infromation gain
def infor_gain(before_split_freqs, after_split_freqs):
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain

#Node calss
class Node(object):
    def __init__(self, l, r, attr, thresh, label=None, active=True):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.label = label
        self.active = active

#Tree class (unused)
class Tree(object):
    def __init__(self, root ):
        pass

#ID3, creates the tree and returns the root
def ID3(train_data, train_labels, max_depth, min_split):
    if len(train_data.columns) == 0 or (train_labels['survived'] == 1).all() == True or (train_labels['survived'] == 0).all() == True or max_depth == 1 or len(train_data) <= min_split :
        if (train_labels['survived'] == 1).all() :
            node = Node(None, None, None, None, 1)
            return node
        elif (train_labels['survived'] == 0).all():
            node = Node(None, None, None ,None, 0)
            return node
        else:
            if (train_labels['survived'] == 1).sum() >= (train_labels['survived'] == 0).sum():
                node = Node(None, None, None, None,  1)
                return node
            else:
                node = Node(None, None, None, None, 0)
                return node
    # 1. use a for loop to calculate the infor-gain of every attribute
    # 1.1 pick a threshold
    # 1.2 split the data using the threshold
    # 1.3 calculate the infor_gain
    # create a df that has labels too
    max_infoGain = []
    comb = pd.concat([train_data, train_labels], axis=1)
    pos_labels = len(comb.loc[comb['survived'] == 1])
    before_split = [len(comb) - pos_labels, pos_labels]
    for att in train_data:
        #train_data[att] = train_data[att].fillna(train_data.mode().iloc[0])
        lab = comb[[att, 'survived']]
        uni = np.unique(lab[att])
        ig_thresh = []
        ig_thresh_as = []

        # if attribute has only 1 unique value
        if len(uni) == 1:
            thresh = uni[0]
            filt_true = len(lab.loc[lab['survived'] == 1])
            after_split = [len(lab) - filt_true, filt_true]
            ig_thresh_as.append(after_split)
            infoG = infor_gain(before_split, ig_thresh_as)
            ig_thresh.append((infoG, thresh))
            ig_thresh_as = []

        for i in (range(len(uni) - 1)):
            thresh = (uni[i] + uni[i + 1]) / 2

            # infoG for right side of threshold
            filt = lab.loc[lab[att] <= thresh]
            if len(filt) != 0:
                filt_true = len(filt.loc[filt['survived'] == 1])
                after_split = [len(filt) - filt_true, filt_true]
                ig_thresh_as.append(after_split)

            # infoG for left side of threshold
            filt = lab.loc[lab[att] > thresh]
            if len(filt) != 0:
            # countnumer of survied in threshold
                filt_true = len(filt.loc[filt['survived'] == 1])
                after_split = [len(filt) - filt_true, filt_true]
                ig_thresh_as.append(after_split)

            # infoGain for attribute
            infoG = infor_gain(before_split, ig_thresh_as)
            ig_thresh.append((infoG, thresh))
            ig_thresh_as = []

        threshold = (max(ig_thresh))
        max_infoGain.append((threshold[0], att, threshold[1]))

    # 2. pick the attribute that achieve the maximum infor-gain
    max_attribute = max(max_infoGain)

    # 3. build a node to hold the data;
    if (train_labels['survived'] == 1).sum() >= (train_labels['survived'] == 0).sum():
        current_node = Node(train_data, train_labels, max_attribute[1], max_attribute[2], label=1)
    else:
        current_node = Node(train_data, train_labels, max_attribute[1], max_attribute[2], label=0)

    # 4. split the data into two parts.
    left = comb.loc[comb[max_attribute[1]] <= max_attribute[2]]
    right = comb.loc[comb[max_attribute[1]] > max_attribute[2]]

    del right[max_attribute[1]]
    del left[max_attribute[1]]

    # 5. call ID3() for the left parts of the data
    left_subtree = ID3(left.iloc[:, :-1], left[['survived']], max_depth-1, min_split)
    # 6. call ID3() for the right parts of the data.
    right_subtree = ID3(right.iloc[:, :-1], right[['survived']], max_depth-1, min_split)

    current_node.left_subtree = left_subtree
    current_node.right_subtree = right_subtree

    return(current_node)

#prints out the tree
def print_tree(root, level = 0):
    print(" " * 4* level, root.attribute, root.threshold, root.label)
    if root.left_subtree is not None:
        print_tree(root.left_subtree, level=level+1)
    if root.right_subtree is not None:
        print_tree(root.right_subtree, level=level+1)

#splits data up into groups to be used in cross fold validation
def k_cross_fold(train_data, train_labels, k):
    comb = pd.concat([train_data, train_labels], axis=1)  # put labels and attributes in 1 df
    shuffled_df = comb.sample(comb.shape[0])
    groups = np.array_split(shuffled_df, k)
    return groups

#get the predicted labels
def predict(root, x):
    preds = []
    for i, row in x.iterrows():
        if root.attribute is None:
            preds.append(root.label)
        elif (row[root.attribute] < root.threshold and root.left_subtree.active):
            preds.extend(predict(root.left_subtree ,x.loc[[i]]))
        elif (row[root.attribute] >= root.threshold and root.right_subtree.active):
            preds.extend(predict(root.right_subtree ,x.loc[[i]]))
        else:
            preds.append(root.label)
    return preds

#get max depth of tree
def max_depth(root):
    if root is None:
        return 0
    lDepth = max_depth(root.left_subtree)
    rDepth = max_depth(root.right_subtree)

    if (lDepth > rDepth):
        return lDepth + 1
    else:
        return rDepth + 1

#get the leaves of the tree
def get_leaves(root):
    if root.active == False:
        return []
    if root.attribute is None or (not root.left_subtree.active and not root.right_subtree.active):
        return [root]
    leaves = []
    if root.left_subtree.active:
        leaves += get_leaves(root.left_subtree)
    if root.right_subtree.active:
        leaves += get_leaves(root.right_subtree)
    return leaves

#prune the tree
def prune(root, x_valid, y_valid, acc_valid):
    n_prunes = 1
    while n_prunes > 0:
        n_prunes = 0
        leaves = get_leaves(root)
        for leaf in leaves:
            leaf.active = False
            preds = predict(root, x_valid)
            acc = (preds==y_valid["survived"]).mean()
            if acc < acc_valid:
                leaf.active = True
            else:
                n_prunes += 1
                acc_valid = acc

def get_active(root):
    if root is None or root.active == False:
        return 0
    return 1+get_active(root.left_subtree)+get_active(root.right_subtree)

#execute k cross fold validation
def perform_kfold(X,Y, k,max_depth, min_split, do_prune):
    folds = k_cross_fold(X,Y,k)
    accs_train = []
    accs_validation = []
    trees = []
    for i in range(k):
        train_idcs = list(range(k))
        train_idcs.remove(i)
        df_train = pd.concat([folds[j] for j in train_idcs])
        df_valid = folds[i]
        x_train = df_train.drop("survived", axis=1)
        y_train = df_train[["survived"]]
        x_valid = df_valid.drop("survived", axis=1)
        y_valid = df_valid[["survived"]]
        root = ID3(x_train, y_train,max_depth, min_split)
        n_active_before = get_active(root)
        preds = predict(root, x_valid)
        #print_tree(root)
        acc = (preds==y_valid["survived"]).mean()
        preds2 = predict(root, x_train)
        acc2 = (preds2==y_train["survived"]).mean()
        if do_prune == True:
            prune(root, x_valid, y_valid, acc)
            n_active_after = get_active(root)
            #change accuracy to get accuracy of post pruned trees
            preds1 = predict(root, x_train)
            acc_after_train = (preds1 == y_train["survived"]).mean()
            acc2 = acc_after_train
            preds = predict(root, x_valid)
            acc_after_valid = (preds == y_valid["survived"]).mean()
            acc = acc_after_valid
        trees.append(root)
        accs_validation.append(acc)
        accs_train.append(acc2)
    #print('tr', accs_train)
    return accs_train, accs_validation, trees




if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder')
    parser.add_argument('--testFolder')
    parser.add_argument('--model')
    parser.add_argument('--crossValidK', type=int, default=5)
    parser.add_argument('--depth', type=int, default=sys.maxsize)
    parser.add_argument('--minSplit', type=int, default=-1)
    args = parser.parse_args()

    if args.model == 'vanilla':
        X = pd.read_csv(args.trainFolder + '/titanic-train.data', delimiter=',', index_col=None, engine='python')
        X = X[X.columns].fillna(X.mode().iloc[0])
        Y = pd.read_csv(args.trainFolder +'/titanic-train.label', delimiter=',', index_col=None, engine='python')
        test_set = pd.read_csv(args.testFolder + '/titanic-test.data', delimiter=',', index_col=None, engine='python')
        test_set = test_set[test_set.columns].fillna(test_set.mode().iloc[0])
        test_label = pd.read_csv(args.testFolder + '/titanic-test.label', delimiter=',', index_col=None, engine='python')
        root = ID3(X, Y, args.depth, args.minSplit)
        preds = predict(root, X)
        accs = perform_kfold(X, Y, args.crossValidK, args.depth, args.minSplit, do_prune=False)
        #print(accs)
        for i in range(args.crossValidK):
            print("fold=" + str(i + 1) + ", train set accuracy=" + str(accs[0][i]) + ", validation set accuracy=" + str(accs[1][i]))

        #majorty voting for test accuracy
        maj_vote = pd.DataFrame(columns=range(len(test_set)))
        for i in (accs[2]):
            vals = predict(i,test_set )
            maj_vote.loc[len(maj_vote)] = vals
        test_pred = maj_vote.mode(axis=0)
        acc2 = (test_pred.iloc[0]==test_label["survived"]).mean()

        print("Test set accuracy= "+ str(acc2))

    if args.model == 'depth':
        X = pd.read_csv(args.trainFolder + '/titanic-train.data', delimiter=',', index_col=None, engine='python')
        X = X[X.columns].fillna(X.mode().iloc[0])
        Y = pd.read_csv(args.trainFolder + '/titanic-train.label', delimiter=',', index_col=None, engine='python')
        test_set = pd.read_csv(args.testFolder + '/titanic-test.data', delimiter=',', index_col=None, engine='python')
        test_set = test_set[test_set.columns].fillna(test_set.mode().iloc[0])
        test_label = pd.read_csv(args.testFolder + '/titanic-test.label', delimiter=',', index_col=None, engine='python')
        root = ID3(X, Y, args.depth, args.minSplit)
        preds = predict(root, X)
        accs = perform_kfold(X, Y, 5, args.depth, args.minSplit, do_prune=False)
        for i in range(args.crossValidK):
            print("fold=" + str(i + 1) + ", train set accuracy=" + str(accs[0][i]) + ", validation set accuracy=" + str(
                accs[1][i]))

        # majorty voting for test accuracy
        maj_vote = pd.DataFrame(columns=range(len(test_set)))
        for i in (accs[2]):
            vals = predict(i, test_set)
            maj_vote.loc[len(maj_vote)] = vals
        test_pred = maj_vote.mode(axis=0)
        acc2 = (test_pred.iloc[0] == test_label["survived"]).mean()


        print("Test set accuracy= " + str(acc2))


    if args.model == 'minSplit':
        X = pd.read_csv(args.trainFolder + '/titanic-train.data', delimiter=',', index_col=None, engine='python')
        X = X[X.columns].fillna(X.mode().iloc[0])
        Y = pd.read_csv(args.trainFolder + '/titanic-train.label', delimiter=',', index_col=None, engine='python')
        test_set = pd.read_csv(args.testFolder + '/titanic-test.data', delimiter=',', index_col=None, engine='python')
        test_set = test_set[test_set.columns].fillna(test_set.mode().iloc[0])
        test_label = pd.read_csv(args.testFolder + '/titanic-test.label', delimiter=',', index_col=None, engine='python')
        root = ID3(X, Y, args.depth, args.minSplit)
        preds = predict(root, X)
        accs = perform_kfold(X, Y, args.crossValidK, args.depth, args.minSplit, do_prune=False)
        for i in range(args.crossValidK):
            print("fold="+ str(i+1) + ", train set accuracy="+ str(accs[0][i]) +", validation set accuracy="+ str(accs[1][i]))

        # majorty voting for test accuracy
        maj_vote = pd.DataFrame(columns=range(len(test_set)))
        for i in (accs[2]):
            vals = predict(i, test_set)
            maj_vote.loc[len(maj_vote)] = vals
        test_pred = maj_vote.mode(axis=0)
        acc2 = (test_pred.iloc[0] == test_label["survived"]).mean()

        print("Test set accuracy= " + str(acc2))

    if args.model == 'postPrune':
        X = pd.read_csv(args.trainFolder + '/titanic-train.data', delimiter=',', index_col=None, engine='python')
        X = X[X.columns].fillna(X.mode().iloc[0])
        Y = pd.read_csv(args.trainFolder + '/titanic-train.label', delimiter=',', index_col=None, engine='python')
        test_set = pd.read_csv(args.testFolder + '/titanic-test.data', delimiter=',', index_col=None, engine='python')
        test_set = test_set[test_set.columns].fillna(test_set.mode().iloc[0])
        test_label = pd.read_csv(args.testFolder + '/titanic-test.label', delimiter=',', index_col=None,engine='python')
        root = ID3(X, Y, args.depth, args.minSplit)
        preds = predict(root, X)
        accs = perform_kfold(X, Y, args.crossValidK, max_depth=args.depth, min_split=args.minSplit, do_prune=True)
        for i in range(args.crossValidK):
            print("fold="+ str(i+1) + ", train set accuracy="+ str(accs[0][i]) +", validation set accuracy="+ str(accs[1][i]))

        # majorty voting for test accuracy
        maj_vote = pd.DataFrame(columns=range(len(test_set)))
        for i in (accs[2]):
            vals = predict(i, test_set)
            maj_vote.loc[len(maj_vote)] = vals
        test_pred = maj_vote.mode(axis=0)
        acc2 = (test_pred.iloc[0] == test_label["survived"]).mean()

        print("Test set accuracy= " + str(acc2))

