# coding: utf-8

# # UCI Adult Dataset Process Demo.
# Only demonstrate the basic useage of this dataset and some algorithms in scikit library(And most copy with minimum changes from [【机器学习实验】scikit-learn的主要模块和基本使用](http://www.jianshu.com/p/1c6efdbce226)), if we want to have better and more meaningful results,we should do some more work.

# ## load the preprocessing: transform description to number([male,female]->[0,1])
def loadXandY(fileName):
    import numpy as np

    # load the CSV file as a numpy matrix
    with open(fileName, 'r') as f:
        dataset = np.loadtxt(f, delimiter=",")

    # separate the data from the target attributes
    attrsize = len(dataset[0])

    print(len(dataset), attrsize)

    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    return X, y




# # deal with the missing value
# [Sci4.3.4. Encoding categorical features](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)


def simpleImputer(X):
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    return imp.transform(X)



# ## The doc mentioned that we should use one hot encoding since `these expect continuous input, and would interpret the categories as being ordered, which is often not desired `.
# 
# See [scikit-learn机器学习RandomForest实例（含类别属性处理）](http://blog.csdn.net/mach_learn/article/details/40428297)
# and [Note on using OneHotEncoder in scikit-learn to work on categorical features](https://xgdgsc.wordpress.com/2015/03/20/note-on-using-onehotencoder-in-scikit-learn-to-work-on-categorical-features/) for more info.
# 
# And our feature is:
# ![features](pics/describe.png)


def oneHotEncoderX(X, n_values, categorical_features='all'):
    import numpy as np
    from sklearn import preprocessing

    #     enc = preprocessing.OneHotEncoder(categorical_features=to_change_features) # if not specify the n_values, by default, test set will be 104 long
    enc = preprocessing.OneHotEncoder(n_values, categorical_features=categorical_features)
    enc.fit(X)
    return np.array(enc.transform(X).toarray())




# 数据归一化(Data Normalization)

def simpleScale(X):
    from sklearn import preprocessing
    # normalize the data attributes
    # normalized_X = preprocessing.normalize(X)
    # standardize the data attributes
    return preprocessing.scale(X)

'''
Copy from `https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/preprocessing/data.py#L1728`

def _transform_selected(X, transform, selected="all", copy=True):

It transform the selected cols and put the unchanged cols to right.
'''
def scaleWithFeatures(X, selected="all",  copy=True):
    import six
    from sklearn.preprocessing import scale
    import numpy as np
    from scipy import sparse

    X = X.copy() # Do not pollute the outside X

    if isinstance(selected, six.string_types) and selected == "all":
        return scale(X)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return scale(X)
    else:
        X_sel = scale(X[:, ind[sel]])
        X_not_sel = X[:, ind[not_sel]]

        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel))
        else:
            return np.hstack((X_sel, X_not_sel))





def feature_selection(mdoel, X, y):
    # 特征选择(Feature Selection)
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(X, y)
    # display the relative importance of each attribute
    print(model.feature_importances_)



def show_predictions(y, predicted):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()



def make_predictions(model, X, y):
    from sklearn import metrics
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print('\n')
#     show_predictions(expected[::len(expected)//20], predicted[::len(predicted)//20])



def cross_validation(model, X, y):
    from sklearn.model_selection import KFold, cross_val_score
    k_fold = KFold(n_splits=3)
    return cross_val_score(model, X, y, cv=k_fold, n_jobs=-1)



def getLine(fileName, lineNum):
    with open(fileName, 'r') as f:
        all = f.readlines()

    return all[lineNum].strip()

# getLine(baseDir + 'adult.data', 848)




def testScaleWithFeatures():
    from sklearn import preprocessing
    import numpy as np
    X = np.array([[1., -1., 2.],
                  [2., 0., 0.],
                  [0., 1., -1.]])

    directScaledX = preprocessing.scale(X)

    scaledX =  scaleWithFeatures(X, 'all')

    firstColScaledX = scaleWithFeatures(X, [0,2])

    print(X)

    print((directScaledX==scaledX).all())

    print(directScaledX)

    print(firstColScaledX)

    print(type(X), type(firstColScaledX))



def getScaledAndOneHotEncoderedX(X):
    onehot_features = list(map(lambda x: x - 1, [2, 4, 6, 7, 8, 9, 10, 14]))
    onehot_n_values = [8, 16, 7, 14, 6, 5, 2, 41]

    continuous_features = list(map(lambda x: x - 1, [1, 3, 5, 11, 12, 13]))

    # Attention, since onehot encoder and scaleWithFeatures both will reshape the arrays, so, we need to take care of the order
    # Suggest scaleWithFeatures first, since there is less cols after change. But then, the onehot features will not remain the orginal order,
    # And becomes range(len(continuous_features)+1, X.shape[1]), thus range(6,14)

    scaleWithFeaturesX = scaleWithFeatures(X, continuous_features)

    onehotEncoderedX = oneHotEncoderX(scaleWithFeaturesX, n_values=onehot_n_values, categorical_features=list(range(6, 14)))
    # print(X[1])
    # print(scaleWithFeaturesX[1])
    # print(len(onehotEncoderedX[1]))

    return onehotEncoderedX


def loadData():
    baseDir = 'H:/practice/scikit_class/scikit_learning/uci_adult/adult_data/'
    # baseDir = 'adult_data/'

    fileName = baseDir + 'adult.data.num'

    testFileName = baseDir + 'adult.test.num'

    X, y = loadXandY(fileName)
    TX, Ty = loadXandY(testFileName)

    X = simpleImputer(X)
    TX = simpleImputer(TX)

    return X,y,TX,Ty


def testModelOnData(model, X, y, TX, Ty):
    model.fit(X, y)
    print(model)

    make_predictions(model, X, y)

    make_predictions(model, TX, Ty)

    print(cross_validation(model, X, y))


def logisticRegression(X, y, TX, Ty):
    # 逻辑回归
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    testModelOnData(model, X, y, TX, Ty)


def decisionTree(X, y, TX, Ty):
    from sklearn.tree import DecisionTreeClassifier
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    testModelOnData(model, X, y, TX, Ty)


def gaussianNBDemo(X, y, TX, Ty):
    # 朴素贝叶斯
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    testModelOnData(model, X, y, TX, Ty)

def kNeighborsDemo(X, y, TX, Ty):
    # K近邻
    from sklearn.neighbors import KNeighborsClassifier
    # fit a k-nearest neighbor model to the data
    model = KNeighborsClassifier()
    testModelOnData(model, X, y, TX, Ty)


def svmDemo(X, y, TX, Ty):
    # SVM
    from sklearn.svm import SVC
    # fit a SVM model to the data
    model = SVC()
    testModelOnData(model, X, y, TX, Ty)

def testWithScaledAndOneHotEncoderedData():

    X, y, TX, Ty = loadData()

    niceX = getScaledAndOneHotEncoderedX(X)
    niceTX = getScaledAndOneHotEncoderedX(TX)

    # 决策树
    # decisionTree(X, y, TX, Ty)
    # decisionTree(niceX, y, niceTX, Ty)

    # 逻辑回归, become better
    # logisticRegression(X, y, TX, Ty) # [ 0.79767828  0.79887599  0.79839676]
    # logisticRegression(niceX, y, niceTX, Ty) # [ 0.84973282  0.85028561  0.84953469]

    # gaussianNB, become worse
    # gaussianNBDemo(X, y, TX, Ty)    # [ 0.79390087  0.79546711  0.79747535]
    # gaussianNBDemo(niceX, y, niceTX, Ty)    # [ 0.47475585  0.54809287  0.47765595]


    # KNN, cost much more time to the transformed dataset, become better
    # kNeighborsDemo(X, y, TX, Ty) # [ 0.77833057  0.7740925   0.7734267 ]
    # kNeighborsDemo(niceX, y, niceTX, Ty) # [ 0.82697623  0.83149069  0.83516079]


    svmDemo(X, y, TX, Ty)
    svmDemo(niceX, y, niceTX, Ty)


if __name__ == '__main__':
    testWithScaledAndOneHotEncoderedData()







    # testScaleWithFeatures()
