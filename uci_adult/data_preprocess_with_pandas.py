# coding: utf-8

import pandas as pd #this is how I usually import pandas

from data_preprocess import get_feature_names


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
!!!!!!!!!!!!
The last column is the category, just do not change it
"""
def oneHotWithPandas(df):
    df = df.iloc[:,:-1].copy()
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html, onehot encoding
    df_with_dummies = pd.get_dummies(df) # if not specify `dummy_na`, then `NaN` is ignored.
    df_with_dummies = pd.DataFrame(df_with_dummies, dtype=int) # to make sure the result is integer
    return df_with_dummies






def encode_categorical_features(data):
    data = data.copy()
    import pandas as pd
    import numpy as np

    # data = pd.read_csv(data_file)
    # data = pd.DataFrame.from_csv(data_file) # see [read_csv_a_little_not_as_thought](http://stackoverflow.com/questions/31620667/why-is-pandas-read-csv-not-the-reciprocal-of-pandas-dataframe-to-csv)

    features_names = data.columns.tolist()

    features_continuous = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features_non_continuous = list(set(features_names).difference(set(features_continuous)))

    print(features_continuous)
    print(features_non_continuous)


    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.factorize.html
    for feature_name in features_non_continuous:
        data[feature_name] = data[feature_name].factorize()[0]

    data = pd.DataFrame(data, dtype=np.int32)
    data = data.replace('-1','NaN',regex=True) # It's trick, really, if we use `np.nan`, which seems nice, but the column will become `float64` instead of int
    return data


if __name__ == '__main__':

    base_dir = 'adult_data/'
    describe_file = base_dir+'describe.txt'
    feature_names = get_feature_names(describe_file)
    feature_names.append('category') # The last line is the real category

    # data_files = [base_dir+'adult.data']#, base_dir+'adult.test']
    data_files = [base_dir+'adult.all']

    for data_file in data_files:

        df, df_noNaN = loadFromFileWithPandas(data_file, feature_names)
        df.to_csv(data_file + '.NaN')
        df_noNaN.to_csv(data_file + '.noNaN')

        """
        For tree base classification algos, just encode them and scale the necessary features
        """
        df_tree = encode_categorical_features(df)
        # these features is have much better maxmum than others, when used in `distance based` algo, can be not so good
        df_tree = scaleDataFrameWithFeatures(df_tree, selected=['fnlwgt', 'capital-gain','capital-loss'])
        df_tree.to_csv(data_file + '.scale')


        # deal with missing value, use mean or other methods


        """
        For distance based algos, it's better to use onehot methods
        This way can have feature info saved, a.k.a, `workclass_Federal-gov,workclass_Local-gov,workclass_Never-worked,workclass_Private,workclass_Self-emp-inc`
        But, it is not so easy to deal with missing values.
        """
        df_with_dummies = oneHotWithPandas(df)
        df_with_dummies = scaleDataFrameWithFeatures(df_with_dummies, selected=['fnlwgt', 'capital-gain','capital-loss'])
        df_with_dummies.to_csv(data_file+'.onehot')