# coding: utf-8

from data_preprocessing import loadData,getScaledAndOneHotEncoderedX,getLineFromFile,simpleScale,testModelOnData
from data_preprocessing import decisionTreeDemo

'''
def decisionTreeDemo(X, y, TX, Ty):
    from sklearn.tree import DecisionTreeClassifier
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    testModelOnData(model, X, y, TX, Ty)
'''

def randomForestDemo(X, y, TX, Ty):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    testModelOnData(model, X, y, TX, Ty)


def extraTreeDemo(X, y, TX, Ty):
    from sklearn.ensemble import ExtraTreesClassifier
    # model = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    model = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2,
                                 max_features=4, random_state=0)
    testModelOnData(model, X, y, TX, Ty)


def threeTreeDemo(X, y, TX, Ty):
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier


    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                                 random_state=0)
    scores = cross_val_score(clf, X, y)
    print(scores, scores.mean())

    clf = RandomForestClassifier(n_estimators=10, max_depth=None,
                                 min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, X, y)
    print(scores, scores.mean())

    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                               min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, X, y)
    print(scores, scores.mean())


if __name__ == '__main__':

    X, y, TX, Ty = loadData()

    niceX = getScaledAndOneHotEncoderedX(X)
    niceTX = getScaledAndOneHotEncoderedX(TX)

    # decisionTreeDemo(X, y, TX, Ty) # [ 0.80864198  0.81297218  0.816272  ]
    # decisionTreeDemo(niceX, y, niceTX, Ty) # [ 0.81352497  0.80882624  0.81811481]

    # randomForestDemo(X, y, TX, Ty) # [ 0.83904551  0.8442049   0.8465862 ]
    # randomForestDemo(niceX, y, niceTX, Ty) # [ 0.84125668  0.84411277  0.83764858]

    # threeTreeDemo(X, y, TX, Ty)
    # threeTreeDemo(niceX, y, niceTX, Ty)

    extraTreeDemo(X, y, TX, Ty) # [ 0.83047724  0.83600516  0.83939924]
    extraTreeDemo(niceX, y, niceTX, Ty) # [ 0.8266077   0.8245808   0.82557818]