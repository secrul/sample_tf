import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1*x_data + 0.3

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))#产生均匀分布的随机张量
b = tf.Variable(tf.zeros([1]))

y = W*x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init) #激活

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(W),sess.run(b))