## 大数据算法课程作业

UCI Adult 数据集

## 数据获取

`data_retrive.py`

## 数据预处理

First, read the article proposed from the scikit tutorial,[4.3.1.3. Scaling data with outliers](http://scikit-learn.org/stable/modules/preprocessing.html#scaling-data-with-outliers): [I have not read it yet -.-]
>Further discussion on the importance of centering and scaling data is available on this FAQ: [Should I normalize/standardize/rescale the data?](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)

### Encoding categorical features(将描述符转化为数字)

`data_preprocess.py`

`sex: Female, Male.` 转化为 `0,1`

从 scikit 摘取一些类似的内容：
[Encoding categorical features](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features):

>Often features are not given as continuous values but categorical. For example a person could have features ["male", "female"], ["from Europe", "from US", "from Asia"], ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]. Such features can be efficiently coded as integers, for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3] while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].


In fact, there are some codes in libs, see [Converting categorical data into numbers with Pandas and Scikit-learn](http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/)

``` python
cols_to_transform = [ 'a', 'list', 'of', 'categorical', 'column', 'names' ]
df_with_dummies = df.get_dummies( columns = cols_to_transform )
```

and [Note on using OneHotEncoder in scikit-learn to work on categorical features](https://xgdgsc.wordpress.com/2015/03/20/note-on-using-onehotencoder-in-scikit-learn-to-work-on-categorical-features/)

``` python
encoder = sklearn.preprocessing.OneHotEncoder()
label_encoder = sklearn.preprocessing.LabelEncoder()
data_label_encoded = label_encoder.fit_transform(data['category_feature'])
data['category_feature'] = data_label_encoded
data_feature_one_hot_encoded = encoder.fit_transform(data[['category_feature']].as_matrix())
```

### Continuous attributes should (normalize and) scale. 

`preprocessing.scale(X)`, NOTE, onehot encoded attrs should not scale.

## TODO
