## 大数据算法课程作业

UCI Adult 数据集

## 数据获取

`data_retrive.py`

## 数据预处理

First, read the article proposed from the scikit tutorial,[4.3.1.3. Scaling data with outliers](http://scikit-learn.org/stable/modules/preprocessing.html#scaling-data-with-outliers): [I have not read it yet -.-]
>Further discussion on the importance of centering and scaling data is available on this FAQ: [Should I normalize/standardize/rescale the data?](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)

**See [data_prpcessing](data_processing.md) for more info**

[`randomized_search_decision_tree`](play_with_adult_data/randomized_search_decision_tree.py)

``` python
"""
Pandas' way to preprocess data
"""
import sys
sys.path.append('..')
from data_preprocess_with_pandas import loadDataFromFileWithPandas

base_dir = '../adult_data/'
data_file = base_dir + 'adult.all.scale'
X_train, X_test, y_train, y_test = loadDataFromFileWithPandas(data_file)
X = X_train,y = y_train,TX = X_test,Ty = y_test
```


## feature selection/dimensionality reduction

We have two types of data,`adult.all.scale` and `adult.all.onehot`, for `Tree based` classification algo, it's just okay to use the former, but for many other algos.

## Choose the proper algorithms and tweak the parameters.




## Conclusion

In fact, for a classification task ,with the `GridSearchCV` and/or `RandomizedSearchCV` to get nice params for specific algorithms, we only need to find the proper algos to use and just concentrate on providing nice data to the algos.

So, preprocessing is essential, the onehotencoding if necessary, feature selection/dimensionality reduction if possible.

And the most important is the ability to explain the results, why the results are getting better and why just cannot continue to be better.