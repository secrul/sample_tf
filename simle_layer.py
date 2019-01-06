import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_fun = None):
    Weights = tf.Variable(tf.random_uniform([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) #不推荐初始值为0
    Wx_plus_biases = tf.matmul(inputs,Weights) + biases
    output = Wx_plus_biases if activation_fun is None else activation_fun(Wx_plus_biases)
    return output

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])#None 表示给任意个example都行
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_fun=tf.nn.relu)

predition = add_layer(l1,10,1,activation_fun=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

se = tf.Session()

se.run(init)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
for i in range(2000):
    se.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predition_value = se.run(predition, feed_dict={xs: x_data})
        lines = ax.plot(x_data,predition_value,'r-',lw = 5)
        plt.pause(0.1)
plt.ioff()
plt.show()
