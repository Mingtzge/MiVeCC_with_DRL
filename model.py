# 定义了单个Agent的DDPG结构，及一些函数

import tensorflow as tf
import tensorflow.contrib as tc
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# from torch import optim
# from torch.autograd import Variable


class MADDPG():
    def __init__(self, name, actor_lr, critic_lr, layer_norm=True, nb_actions=300,
                 num_units=256, state_len=4):
        # nb_input = state_len * nb_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_input = tf.placeholder(shape=[None, self.nb_actions, self.nb_actions, 3], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, self.nb_actions], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # 输入是一个具体的状态state，经过两层的全连接网络输出选择的动作action
        def actor_network(name, state_input, num_action):
            with tf.variable_scope(name) as scope:
                x = state_input
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                # conv1 7*7*32
                # layers.conv2d parameters
                # inputs 输入，是一个张量
                # filters 卷积核个数，也就是卷积层的厚度
                # kernel_size 卷积核的尺寸
                # strides: 扫描步长
                # padding: 边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
                # activation: 激活函数
                conv1 = tf.layers.conv2d(
                    inputs=x,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.tanh
                )
                # pool1 36*36*64
                # tf.layers.max_pooling2d
                # inputs 输入，张量必须要有四个维度
                # pool_size: 过滤器的尺寸

                # pool1 = tf.layers.max_pooling2d(
                #     inputs=conv1,
                #     pool_size=[2, 2],
                #     strides=2
                # )
                # conv1 14*14*32
                # layers.conv2d parameters
                # inputs 输入，是一个张量
                # filters 卷积核个数，也就是卷积层的厚度
                # kernel_size 卷积核的尺寸
                # strides: 扫描步长
                # padding: 边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
                # activation: 激活函数
                conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=16,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.tanh
                )
                # pool1 7*7*32
                # tf.layers.max_pooling2d
                # inputs 输入，张量必须要有四个维度
                # pool_size: 过滤器的尺寸

                # pool2 = tf.layers.max_pooling2d(
                #     inputs=conv2,
                #     pool_size=[2, 2],
                #     strides=2
                # )

                # flat(平坦化)
                flat = tf.reshape(conv2, [-1, ((num_action - 3) // 4) * ((num_action - 3) // 4) * 16])
                # 形状变成了[?,1568]
                if self.layer_norm:
                    flat = tc.layers.layer_norm(flat, center=True, scale=True)
                x = tf.layers.dense(flat, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.tanh(x)

                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))  # 全连接层
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.tanh(x)

                # x = tf.layers.dense(x, num_action,
                #                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))  # 全连接层
                # if self.layer_norm:
                #     x = tc.layers.layer_norm(x, center=True, scale=True)
                # x = tf.nn.relu(x)

                x = tf.layers.dense(x, num_action,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                # x = tf.nn.softmax(x)
                # x = tf.arg_max(x, 1)
                # x = tf.cast(tf.reshape(x, [-1, 1]), dtype=tf.float32)
                # bias = tf.constant(-30, dtype=tf.float32)
                w_ = tf.constant(3, dtype=tf.float32)
                # x = tf.multiply(tf.add(x, bias), w_)
                x = tf.multiply(tf.nn.tanh(x), w_)
            return x

        # 输入时 state，所有Agent当前的action信息
        def critic_network(name, state_input, action_input, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                x = state_input
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                # conv1 72*72*64
                # layers.conv2d parameters
                # inputs 输入，是一个张量
                # filters 卷积核个数，也就是卷积层的厚度
                # kernel_size 卷积核的尺寸
                # strides: 扫描步长
                # padding: 边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
                # activation: 激活函数
                conv1 = tf.layers.conv2d(
                    inputs=x,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.relu
                )
                # pool1 36*36*64
                # tf.layers.max_pooling2d
                # inputs 输入，张量必须要有四个维度
                # pool_size: 过滤器的尺寸

                # pool1 = tf.layers.max_pooling2d(
                #     inputs=conv1,
                #     pool_size=[2, 2],
                #     strides=2
                # )
                # conv1 14*14*32
                # layers.conv2d parameters
                # inputs 输入，是一个张量
                # filters 卷积核个数，也就是卷积层的厚度
                # kernel_size 卷积核的尺寸
                # strides: 扫描步长
                # padding: 边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
                # activation: 激活函数
                conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=16,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.relu
                )
                # pool1 7*7*32
                # tf.layers.max_pooling2d
                # inputs 输入，张量必须要有四个维度
                # pool_size: 过滤器的尺寸

                # pool2 = tf.layers.max_pooling2d(
                #     inputs=conv2,
                #     pool_size=[2, 2],
                #     strides=2
                # )

                # flat(平坦化)
                flat = tf.reshape(conv2,
                                  [-1, ((action_input.shape[1] - 3) // 4) * ((action_input.shape[1] - 3) // 4) * 16])
                # 形状变成了[?,1568]
                x = tf.concat([flat, action_input], axis=-1)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                # x = tf.concat([x, action_input], axis=-1)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, num_units / 2,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, num_units / 4,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            return x

        self.state_input = state_input
        self.action_input = action_input
        self.reward = reward
        self.action_output = actor_network(name + "_actor", state_input=self.state_input, num_action=self.nb_actions)
        self.critic_output = critic_network(name + '_critic',
                                            action_input=self.action_input, state_input=self.state_input)

        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        # 最大化Q值
        self.actor_loss = -tf.reduce_mean(
            critic_network(name + '_critic',
                           action_input=self.action_output,
                           reuse=True, state_input=self.state_input))  # reduce_mean 为求均值，即为期望
        online_var = [i for i in tf.trainable_variables() if name + "_actor" in i.name]
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss, var_list=online_var)
        # self.actor_train = self.actor_optimizer.minimize(self.actor_loss)
        self.actor_loss_op = tf.summary.scalar("actor_loss", self.actor_loss)
        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))  # 目标Q 与 真实Q 之间差的平方的均值
        self.critic_loss_op = tf.summary.scalar("critic_loss", self.critic_loss)
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)
        self.count = 0

    def train_actor(self, state, action, sess, summary_writer, lr):
        self.count += 1
        self.actor_lr = lr
        summary_writer.add_summary(
            sess.run(self.actor_loss_op, {self.state_input: state, self.action_input: action}), self.count)
        sess.run(self.actor_train, {self.state_input: state, self.action_input: action})

    def train_critic(self, state, action, target, sess, summary_writer, lr):
        self.critic_lr = lr
        summary_writer.add_summary(
            sess.run(self.critic_loss_op, {self.state_input: state, self.action_input: action,
                                           self.target_Q: target}), self.count)
        sess.run(self.critic_train,
                 {self.state_input: state, self.action_input: action, self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, sess):
        return sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action})


class M_MADDPG():
    def __init__(self, name, actor_lr, critic_lr, layer_norm=True, nb_actions=300,
                 num_units=256, state_len=4):
        # nb_input = state_len * nb_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_input = tf.placeholder(shape=[None, self.nb_actions, self.nb_actions, 3], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, self.nb_actions], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # 输入是一个具体的状态state，经过两层的全连接网络输出选择的动作action
        def actor_network(name, state_input, num_action):
            with tf.variable_scope(name) as scope:
                x = state_input
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                # conv1 7*7*32
                # layers.conv2d parameters
                # inputs 输入，是一个张量
                # filters 卷积核个数，也就是卷积层的厚度
                # kernel_size 卷积核的尺寸
                # strides: 扫描步长
                # padding: 边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
                # activation: 激活函数
                # input 60*60*3
                # output 29*29*32
                conv1 = tf.layers.conv2d(
                    inputs=x,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.tanh
                )
                # input 29*29*32
                # output 14*14*32
                conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.tanh
                )
                # input 14*14*32
                # output 6*6*16
                conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=16,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.tanh
                )
                # flat(平坦化)
                flat = tf.reshape(conv3, [-1, 6 * 6 * 16])
                # 形状变成了[?,1568]
                if self.layer_norm:
                    flat = tc.layers.layer_norm(flat, center=True, scale=True)
                x = tf.layers.dense(flat, num_units*2,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.tanh(x)

                x = tf.nn.tanh(x)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))  # 全连接层
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.tanh(x)
                x = tf.layers.dense(x, num_action,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

                w_ = tf.constant(3, dtype=tf.float32)
                # x = tf.multiply(tf.add(x, bias), w_)
                x = tf.multiply(tf.nn.tanh(x), w_)
            return x

        # 输入时 state，所有Agent当前的action信息
        def critic_network(name, state_input, action_input, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                x = state_input
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                conv1 = tf.layers.conv2d(
                    inputs=x,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.relu
                )
                # input 30*30*32
                # output 15*15*32
                conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=32,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.relu
                )
                # input 15*15*32
                # output 7*7*16
                conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=16,
                    kernel_size=[3, 3],
                    strides=2,
                    padding="valid",
                    activation=tf.nn.relu
                )
                # flat(平坦化)
                flat = tf.reshape(conv3, [-1, 6 * 6 * 16])
                x = tf.concat([flat, action_input], axis=-1)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.layers.dense(x, num_units*2,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                # x = tf.concat([x, action_input], axis=-1)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, num_units / 4,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            return x

        self.state_input = state_input
        self.action_input = action_input
        self.reward = reward
        self.action_output = actor_network(name + "_actor", state_input=self.state_input, num_action=self.nb_actions)
        self.critic_output = critic_network(name + '_critic',
                                            action_input=self.action_input, state_input=self.state_input)

        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        # 最大化Q值
        self.actor_loss = -tf.reduce_mean(
            critic_network(name + '_critic',
                           action_input=self.action_output,
                           reuse=True, state_input=self.state_input))  # reduce_mean 为求均值，即为期望
        online_var = [i for i in tf.trainable_variables() if name + "_actor" in i.name]
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss, var_list=online_var)
        # self.actor_train = self.actor_optimizer.minimize(self.actor_loss)
        self.actor_loss_op = tf.summary.scalar("actor_loss", self.actor_loss)
        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))  # 目标Q 与 真实Q 之间差的平方的均值
        self.critic_loss_op = tf.summary.scalar("critic_loss", self.critic_loss)
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)
        self.count = 0

    def train_actor(self, state, action, sess, summary_writer, lr):
        self.count += 1
        self.actor_lr = lr
        summary_writer.add_summary(
            sess.run(self.actor_loss_op, {self.state_input: state, self.action_input: action}), self.count)
        sess.run(self.actor_train, {self.state_input: state, self.action_input: action})

    def train_critic(self, state, action, target, sess, summary_writer, lr):
        self.critic_lr = lr
        summary_writer.add_summary(
            sess.run(self.critic_loss_op, {self.state_input: state, self.action_input: action,
                                           self.target_Q: target}), self.count)
        sess.run(self.critic_train,
                 {self.state_input: state, self.action_input: action, self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, sess):
        return sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action})
