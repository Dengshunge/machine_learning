# machine_learning
Machine_Learning文件夹的代码主要是《机器学习实战》，我将其转化为了3.6版本  
Summary文件夹是我在学习机器学习中，对不同算法的一个总结

## L1、L2正则化的作用
1. L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
2. L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合  
具体解释可以见[这篇文章](https://blog.csdn.net/jinping_shi/article/details/52433975)  

另外[这篇文章](https://blog.csdn.net/crazy_scott/article/details/80343324)用另外的方法介绍了两者的区别和联系
![](https://github.com/Dengshunge/machine_learning/blob/master/buffer/L1L2%E6%AD%A3%E5%88%99%E5%8C%96%E7%9A%84%E8%81%94%E7%B3%BB.jpg?raw=true)

## 归一化
[为什么一些机器学习模型需要对数据进行归一化？](https://www.cnblogs.com/bonelee/p/7124695.html)  
1. 提高收敛速度，若不采用归一化，模型很难收敛甚至不收敛。
2. 归一化有可能提高模型的精度。
3. 归一化的类型:1.线性归一化（即减去最小值然后除以范围）（适合数据比较集中的情况）；2.z分数归一化（即0均值）（经过处理的数据符合正太分布）；3.非线性归一化（利用数学函数进行映射）（适合数据分化比较大的场景）

## 防止过拟合的方法
[感谢这位博主](https://blog.csdn.net/Left_Think/article/details/77684087?locationNum=5&fps=1)  
防止过拟合的方法有：
1. 引入正则化
2. dropout（常用于神经网络，随机抛弃部分神经元）
3. 提前终止训练
4. 增加样本数量

## 交叉验证
[为什么要用交叉验证](https://blog.csdn.net/aliceyangxi1987/article/details/73532651)
