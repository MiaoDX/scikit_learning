# coding: utf-8

import pandas as pd

def get_feature_names(feature_names_file):
    feature_names = []
    with open(feature_names_file, 'r') as f:
        for line in f.readlines():
            feature_name = line[:line.find(':')]
            feature_names.append(feature_name)
    return feature_names


def scaleDataFrameWithFeatures(X, selected=[]):
    X = X.copy()  # Do not pollute the outside X
    from sklearn.preprocessing import scale
    X[selected] = scale(X[selected])
    return X



def loadFromFileWithPandas(fileName, feature_names):
    df = pd.read_csv(fileName, names=feature_names, sep=',\s*')  # use a sep with blank so that we can make sure every item do not have blank before or after
    import numpy as np
    df = df.replace(r'\?', np.nan, regex=True)  # replace the `?` as NaN, to better deal missing values

    df_less = df.dropna() # drop Nan

    print(df.shape, df_less.shape, df.shape[0] - df_less.shape[0])
    return df, df_less



"""
Use pandas get_dummies to do onehot encoder
"""
def oneHotWithPandas(df, features_non_continuous_without_category):
    import numpy as np
    df = df.iloc[:,:-1].copy()
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html, onehot encoding
    df_with_dummies = pd.get_dummies(df,columns=features_non_continuous_without_category) # if not specify `dummy_na`, then `NaN` is ignored.
    df_with_dummies = pd.DataFrame(df_with_dummies, dtype=np.int32) # to make sure the result is integer
    return df_with_dummies


def encode_categorical_features(data,features_non_continuous):
    data = data.copy()
    import numpy as np

    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.factorize.html
    for feature_name in features_non_continuous:
        data[feature_name] = data[feature_name].factorize()[0]

    data = data.replace(-1,np.nan) # we change -1 back to nan to avoid possibly wrong use of -1. However, data type will change to float.
    return data


def loadDataFromFileWithPandas(data_file):
    """

    :param data_file:
    :return:
    """
    from sklearn.model_selection import train_test_split
    df = pd.DataFrame.from_csv(data_file)
    data = df.as_matrix()[:, 0:-1]
    target = df.as_matrix()[:, -1]


    X_train, X_test, y_train, y_test =  train_test_split(data, target, test_size=1.0/4, random_state=0)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def loadDataFromFileWithPandas2(data_file):
    """

    :param data_file:
    :return:
    """
    from sklearn.model_selection import train_test_split
    df = pd.DataFrame.from_csv(data_file)
    data = df.as_matrix()[:, 0:-1]
    target = df.as_matrix()[:, -1]

    X_train = data[:32561]
    y_train = target[:32561]
    X_test = data[32561:]
    y_test = target[32561:]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def preprocessing():

    """
    The result files:

    .all -> combine of data and test

    .all.NaN -> DataFrame format
    .all.noNaN -> drop all NaN value
    .all.fillna -> fill all NaN value
    .all.cate -> encode categorical features
    .all.onehot -> onehot encode of .all.fillna

    .all.onehot.direct -> onehot encode of .all.NaN
    """

    """
    0.Load Raw data and change data format to DataFrame
    """
    base_dir = 'adult_data/'
    describe_file = base_dir+'describe.txt'
    feature_names = get_feature_names(describe_file)
    feature_names.append('category') # The last line is the real category

    # data_files = [base_dir+'adult.data']#, base_dir+'adult.test']
    data_file = base_dir+'adult.all'

    df, df_noNaN = loadFromFileWithPandas(data_file, feature_names)
    df.to_csv(data_file + '.NaN')
    df_noNaN.to_csv(data_file + '.noNaN')


    # get feature names for scale or encodering
    features_names = df.columns.tolist()
    features_continuous = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features_non_continuous = list(set(features_names).difference(set(features_continuous)))
    print(features_continuous)
    print(features_non_continuous)


    """
    2.1 For tree base classification algos, just encode them and scale the necessary features
    """
    df_tree = encode_categorical_features(df, features_non_continuous)
    df_tree.to_csv(data_file + '.cate')


    """
    deal with missing value, use mean or other methods
    """
    import numpy as np
    df_fillna = df_tree.fillna(df_tree.mean())
    df_fillna = pd.DataFrame(df_fillna, dtype=np.int32)
    df_fillna.to_csv(data_file + '.fillna')

    # these features is have much better maxmum than others, when used in `distance based` algo, can be not so good
    df_tree_scale = scaleDataFrameWithFeatures(df_fillna, selected=['fnlwgt', 'capital-gain','capital-loss'])
    df_tree_scale.to_csv(data_file + '.scale')



    """
    2.2 For distance based algos, it's better to use onehot methods
    """
    features_non_continuous_without_category = features_non_continuous.copy()
    features_non_continuous_without_category.remove('category')

    """
    2.2.1 This way can have feature info saved, a.k.a, `workclass_Federal-gov,workclass_Local-gov,workclass_Never-worked,workclass_Private,workclass_Self-emp-inc`
    But, it is not so easy to deal with missing values.
    """
    df_with_dummies = oneHotWithPandas(df, features_non_continuous_without_category) # Note, input is df, without any process
    df_with_dummies = scaleDataFrameWithFeatures(df_with_dummies, selected=['fnlwgt', 'capital-gain','capital-loss'])
    df_with_dummies.to_csv(data_file+'.onehot.direct')


    """
    Almost the same as above, expect that the input data is df_tree_mean, aka, replace `NaN` with mean value and make sure dtype is int32
    The results is alike, but the first line (features names) are different, `workclass_1, workclass_2, workclass_3, workclass_4`
    """
    df_with_dummies2 = oneHotWithPandas(df_fillna, features_non_continuous_without_category)
    df_with_dummies2.to_csv(data_file+'.onehot')

if __name__ == '__main__':
    preprocessing()

    base_dir = 'adult_data/'
    data_file = base_dir + 'adult.all.scale'
    X_train, X_test, y_train, y_test = loadDataFromFileWithPandas(data_file)
