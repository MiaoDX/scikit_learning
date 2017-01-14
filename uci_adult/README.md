## 大数据算法课程作业

UCI Adult 数据集

## 数据获取

`data_retrive.py`

## 数据预处理

First, read the article proposed from the scikit tutorial,[4.3.1.3. Scaling data with outliers](http://scikit-learn.org/stable/modules/preprocessing.html#scaling-data-with-outliers): [I have not read it yet -.-]
>Further discussion on the importance of centering and scaling data is available on this FAQ: [Should I normalize/standardize/rescale the data?](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)



### Continuous attributes should (normalize and) scale. 

`preprocessing.scale(X)`, NOTE, onehot encoded attrs should not scale.

## TODO


## Conclusion

In fact, for a classification task ,with the `GridSearchCV` and/or `RandomizedSearchCV` to get nice params for specific algorithms, we only need to find the proper algos to use and just concentrate on providing nice data to the algos.

So, preprocessing is essential, the onehotencoding if necessary, feature selection/dimensionality reduction if possible.

And the most important is the ability to explain the results, why the results are getting better and why just cannot continue to be better.