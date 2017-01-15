## 大数据算法课程作业

UCI Adult 数据集

## Obtain Data

`data_retrive.py` or just `wget` is enough.

## Data preprocessing

First, read the article proposed from the scikit tutorial,[4.3.1.3. Scaling data with outliers](http://scikit-learn.org/stable/modules/preprocessing.html#scaling-data-with-outliers): [I have not read it yet -.-]

>Further discussion on the importance of centering and scaling data is available on this FAQ: [Should I normalize/standardize/rescale the data?](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)

**See [data_preprocessing.md](data_preprocessing.md) for more info**

When use the data, see [`randomized_search_decision_tree`](play_with_adult_data/randomized_search_decision_tree.py) for a demo.

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
```


## feature selection/dimensionality reduction

We have two types of data, `adult.all.scale` and `adult.all.onehot`, for `Tree based` classification algo, it's just okay to use the former, but for many other algos aka `Distance based` ones use the latter one and do feature selection.

In `sklearn`, there are many functions can do the job, see [1.13. Feature selection](http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection) for more info. They are easy to use but in fact, maybe should have another post about it.

## Choose the proper algorithms and tweak the parameters.

Hand tweak params is just not so appealing, luckily, `sklearn` dose provide nice choices:

`GridSearchCV` and `RandomizedSearchCV`, see [Comparing randomized search and grid search for hyperparameter estimation](http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#sphx-glr-auto-examples-model-selection-randomized-search-py). All we need is specify parameters and distributions to sample from and let our PC to do the dirty job.


## Conclusion

In fact, for a classification task ,with the `GridSearchCV` and/or `RandomizedSearchCV` to get nice params for specific algorithms, we only need to find the proper algos to use and just concentrate on providing nice data to the algos.

So, preprocessing is essential, the onehotencoding if necessary, feature selection/dimensionality reduction if possible.

And the most important is the ability to explain the results, why the results are getting better and why just cannot continue to be better.