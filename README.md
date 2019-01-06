# sample_tf
部分示例参考学习自B站 莫烦 Python 课程 https://www.bilibili.com/video/av16001891/
一些tensorflow入门的示例程序
一：GradientDescentOptimizer.py
  实现了最简单的梯度下降；随机生成x_data,y_data = 0.1*x_data + 0.3;然后用梯度下降法预测权重W和便宜b,学习步长为0.5,学习200次

二：sample_layer.py
    非线性激励函数；实现了自己添加神经网络层，并进行并可视化；
    构造层：
      参数：inputs 输入数据；in_size : 输入数据的size；out_size:输出数据的size;activation_fun 激励函数
       如果激励函数不为None，那么将输入数据传入激励函数，"将线性数据掰弯成非线性"；如果激励函数是None，数据数据不做改变
    数据：x_data是-1到1之间线性数字，y_data = x_data ^ 2 - 0.5 + noise; 并不是确切的二次关系；
      同样采用梯度下降法学习2000次，同时用plt进行可视化；
    效果：可以看到模拟函数不断自我完善逼近真实数据；
    同时构造了一个graph来表示具体的流程，程序运行完之后在pycharm的terminal中运行：tensorboard --logdir=log 语句，将连接复制到浏览器就可以查看graph

