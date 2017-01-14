One of the most important thing to use machine(/deep) learning algorithms is data preprocessing (at least I think so -.-)

With python, we can have a take-home way to simple and useful data preprocessing. See the below pic for a quick glance.

![data_preprocessing](data_preprocessing.svg)

Here we use [`sklearn`](http://scikit-learn.org/) and [`pandas`](http://pandas.pydata.org/pandas-docs/stable/) to do the job. To show with a real world example, we use [`UCI adult database`](http://archive.ics.uci.edu/ml/datasets/Adult) to show the usage.

## 1.Download the database

we can use `wget` or just any way to download the database. As for this database, please download all files include, since apart from the `adult.data` and `adult.test`, the other files includes the info of the dataset, and we will use them below.

## 2.Load data to pandas dataframe

The original data format is Ok, but not so appealing, since we have no idea each column is which feature. And dataframe is a nice data format with feature names in the first line.

And the original dataset have split the dataset into two files, one for training, the other for test. Seems nice, but in fact, maybe we should combine them together and split after preprocessing. We will cover it below.


`adult.test` delete the first line `|1x3 Cross validator`,last column have `.` in the last, remove it:
`sed 's/\.$//' adult.test > adult.test.new`.
And then `cat adult.data adult.test.new > adult.all` is enough,


Load data into dataframe is just read from file:

``` python
def loadFromFileWithPandas(fileName, feature_names):
    df = pd.read_csv(fileName, names=feature_names, sep=',\s*')  # use a sep with blank so that we can make sure every item do not have blank before or after
    import numpy as np
    df = df.replace(r'\?', np.nan, regex=True)  # replace the `?` as NaN, to better deal missing values

    df_less = df.dropna() # drop Nan

    print(df.shape, df_less.shape, df.shape[0] - df_less.shape[0])
    return df, df_less
```

We should note that `feature_names` is not provided in the data file(s), ``old.adult.names` have the `7. Attribute Information:` (adult.names` have similar info), we can use code to get them(or just copy and paste):

`['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']`, there are 14 features.

In the code above, when `read_csv`, since our data file is separated with `, `(with a blank), so use `sep=',\s*'` when read file.

And `?` in the file represents for No value, in the python computation world, `np.nan` is for that use.

The `dropna` will drop raws(samples) with `NaN`, and the results is nicer dataset (with no missing value), but when deal with real world data, we cannot always just depend on drop the missing value.

Save the results to files:

``` python
data_file = 'adult/adult.all'
df, df_noNaN = loadFromFileWithPandas(data_file, feature_names)
df.to_csv(data_file + '.NaN')
df_noNaN.to_csv(data_file + '.noNaN')
```

We got `(48842, 15) (45222, 15) 3620` in the console, which says that, all dataset is 48842 samples with 3620 have `NaN`, which is 7.412%, so just drop them just need better reasons.

So far, we have changed two data file into one and the data format has become DataFrame. Just have a look at the data file:



```
$ head -n 3 adult.all
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K


$ head -n 3 adult.all.NaN
,age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,category
0,39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States,<=50K
1,50,Self-emp-not-inc,83311,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States,<=50K
```

That's it, the first line is the feature names(the last `category` stands for the class).


## 3.Encoding categorical features(将描述符转化为数字)

绝大多数库都需要输入是数字,所以需要将类似于 `sex: Female, Male.` 转化为 `0,1` 这种形式。

从 scikit 摘取一些类似的内容：
[Encoding categorical features](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features):

>Often features are not given as continuous values but categorical. For example a person could have features ["male", "female"], ["from Europe", "from US", "from Asia"], ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]. Such features can be efficiently coded as integers, for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3] while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].


Both of sklearn and pandas have similar function, `LabelEncoder` and `factorize` respectively.

To show in a simple example:

``` python
# Copy from http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder

>>> from sklearn import preprocessing
>>> s = ["paris", "paris", "tokyo", "amsterdam"]
>>> le = preprocessing.LabelEncoder()
>>> le.fit(s)
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(s) 
array([1, 1, 2, 0], dtype=int64)
>>> list(le.inverse_transform([1, 1, 2, 0]))
['paris', 'paris', 'tokyo', 'amsterdam']


# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.factorize.html
>>> import pandas as pd
>>> s_s = pd.Series(s)
>>> s_s.factorize()[0]
array([0, 0, 1, 2])
```

Clearly, both is okay, even the results are a little different, and `LabelEncoder` seems better with `inverse_transform`. However, when the input have `NaN` values (which is common in our use case), `factorize` can deal with `na_sentinel` param(default set -1 when data is `None`):

``` python
>>> s.append(None)
>>> s
['paris', 'paris', 'tokyo', 'amsterdam', None]
>>> le.fit(s)
[...]
TypeError: unorderable types: NoneType() > str()



>>> s_s = pd.Series(s)
>>> s_s.factorize()[0]
array([ 0,  0,  1,  2, -1])

```

Well, so, maybe `sklearn` really did not mean to deal with the `None` or `NaN` values properly.

Back to our adult dataset:

``` python
import pandas as pd
data = pd.DataFrame.from_csv('adult_data/adult.all.NaN')

features_names = data.columns.tolist()

features_continuous = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_non_continuous = list(set(features_names).difference(set(features_continuous)))
print(features_continuous)
print(features_non_continuous)

# features_non_continuous is just categorical features
for feature_name in features_non_continuous:
    data[feature_name] = data[feature_name].factorize()[0]

data = data.replace(-1,np.nan) # we change -1 back to nan to avoid possibly wrong use of -1. However, data type will change to float.
```

Note that, we hard code the `features_continuous` to show continuous features, maybe seems a little not so clean, but no better ways exists (as far as I know).

Well, that's it. For now, we encode the categorical features into 0,1,2,3..., so many classification algos can just use the data (Maybe do some scale and should pay attention to the `NaN` values).


## 4.OneHot Encoder

>Such integer representation can not be used directly with scikit-learn estimators, as these expect continuous input, and would interpret the categories as being ordered, which is often not desired (i.e. the set of browsers was ordered arbitrarily).
One possibility to convert categorical features to features that can be used with scikit-learn estimators is to use a one-of-K or one-hot encoding, which is implemented in OneHotEncoder. This estimator transforms each categorical feature with m possible values into m binary features, with only one active.

To do it with libs, baidued/googled found some nice post:

* [Converting categorical data into numbers with Pandas and Scikit-learn](http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/)

``` python
cols_to_transform = [ 'a', 'list', 'of', 'categorical', 'column', 'names' ]
df_with_dummies = df.get_dummies( columns = cols_to_transform )
```

* and [Note on using OneHotEncoder in scikit-learn to work on categorical features](https://xgdgsc.wordpress.com/2015/03/20/note-on-using-onehotencoder-in-scikit-learn-to-work-on-categorical-features/):

``` python
encoder = sklearn.preprocessing.OneHotEncoder()
label_encoder = sklearn.preprocessing.LabelEncoder()
data_label_encoded = label_encoder.fit_transform(data['category_feature'])
data['category_feature'] = data_label_encoded
data_feature_one_hot_encoded = encoder.fit_transform(data[['category_feature']].as_matrix())
```


Note that `OneHotEncoder` still can not deal with `None` values, and `get_dummies` can:

``` python
data_to_transform = [0,0.0,2,-1,np.nan,None]
dummies = pd.get_dummies(data_to_transform)
print(dummies)
```

we got:

``` vi
   -1.0   0.0   2.0
0   0.0   1.0   0.0
1   0.0   1.0   0.0
2   0.0   0.0   1.0
3   1.0   0.0   0.0
4   0.0   0.0   0.0
5   0.0   0.0   0.0
```

So, `get_dummies` can successfully treat `0` and `0.0` as the same, but the `-1` is not the same as `None` or `np.nan` even in `get_dummies`, so after `factorize` we should replace `-1` with `np.nan` (see above).

However, at this time, we should just deal with the missing value before we use onehot encoder:

``` python
data_to_transform_p = pd.Series([0,0,1.0,2,np.nan,None])
print(data_to_transform_p.mode()[0])
print(data_to_transform_p.mean())
print(data_to_transform_p)
data_to_transform_fillna = data_to_transform_p.fillna(data_to_transform_p.mean())
print(data_to_transform_fillna)
```

Then we got:

``` vi
0.0
0.75
0    0.0
1    0.0
2    1.0
3    2.0
4    NaN
5    NaN
dtype: float64
0    0.00
1    0.00
2    1.00
3    2.00
4    0.75
5    0.75
```

Then, we want to use onehot encode, before that, please change data type to `int32`:

`data_to_transform_fillna = pd.Series(data_to_transform_fillna, dtype=np.int32)`

Now data is:

``` vi
0    0
1    0
2    1
3    2
4    0
5    0
dtype: int32
```

And now, we can use both method of `get_dummies` and `OneHotEncoder`:

``` python
dummies = pd.get_dummies(data_to_transform_fillna)
print(dummies)
```

``` vi
    -1    0    2
0  0.0  1.0  0.0
1  0.0  1.0  0.0
2  0.0  0.0  1.0
3  1.0  0.0  0.0
4  0.0  1.0  0.0
5  0.0  1.0  0.0
```

As for `OneHotencoder`:

``` python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
data_to_transform_fillna_reshape = data_to_transform_fillna.reshape(-1, 1)
enc.fit(data_to_transform_fillna_reshape)
enc.transform(data_to_transform_fillna_reshape).toarray()
```

result is:

``` vi
array([[ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.]])
```

Then they are just the same (as expected).

**So, we replace the missing value with mean value, in fact, that just make not so much sense. However, we really need a method to make things go through.**


Back to adult dataset, we stick to pandas way:

``` python
def oneHotWithPandas(df, features_non_continuous_without_category):
    import numpy as np
    df = df.iloc[:,:-1].copy()
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html, onehot encoding
    df_with_dummies = pd.get_dummies(df,columns=features_non_continuous_without_category) # if not specify `dummy_na`, then `NaN` is ignored.
    df_with_dummies = pd.DataFrame(df_with_dummies, dtype=np.int32) # to make sure the result is integer
    return df_with_dummies
```

`features_non_continuous_without_category` stands for all the categorical features expect `category/class`.

We can use `OneHotEncoder` of course, with some tweaks:

``` vi
# copy from http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder

categorical_features : “all” or array of indices or mask

    Specify what features are treated as categorical.

        ‘all’ (default): All features are treated as categorical.
        array of indices: Array of categorical feature indices.
        mask: Array of length n_features and with dtype=bool.

    Non-categorical features are always stacked to the right of the matrix.
```

As for our use case, just hand count the categorical features:

`[1, 3, 5, 6, 7, 8, 9, 13]`, code snipes can be:

``` python
from sklearn import preprocessing

encoder = preprocessing.OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
encoder.fit(df_tree_mean)
dummies2 = encoder.transform(df_tree_mean).toarray()
dummies2.shape
```

Got `(48842, 106)`, seems okay.

## 5. All in one

I think it's really mess if you go through this post, so just all in one, see [`data_preprocess_with_pandas`](data_preprocess_with_pandas.py).