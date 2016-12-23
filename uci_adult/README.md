## 大数据算法课程作业

UCI Adult 数据集

## 数据获取

`data_retrive.py`

## 数据预处理

### 将描述符转化为数字

`data_preprocess.py`

`sex: Female, Male.` 转化为 `0,1`

从 scikit 摘取一些类似的内容：
[Encoding categorical features](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features):

>Often features are not given as continuous values but categorical. For example a person could have features ["male", "female"], ["from Europe", "from US", "from Asia"], ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]. Such features can be efficiently coded as integers, for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3] while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].

## TODO
