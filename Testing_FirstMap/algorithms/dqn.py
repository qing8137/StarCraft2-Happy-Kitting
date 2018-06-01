import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class DeepQNetwork(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.merged_summary = tf.summary.merge_all()
        #self.writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.cost_his = []
        self.reward = []
        self.memory_counter = 0

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # hidden layer 1
            e_z1 = tf.layers.dense(self.s, 6, activation=None, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='e1')
            tf.summary.histogram('e_z1', e_z1)

            e_bn1 = tf.layers.batch_normalization(e_z1, training=True)
            tf.summary.histogram('e_bn1', e_bn1)

            e_a1 = tf.nn.tanh(e_bn1)
            tf.summary.histogram('e_a1', e_a1)

            # hidden layer 2
            e_z2 = tf.layers.dense(e_a1, 6, activation=None, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e2')
            tf.summary.histogram('e_z2', e_z2)

            e_bn2 = tf.layers.batch_normalization(e_z2, training=True)
            tf.summary.histogram('e_bn2', e_bn2)

            e_a2 = tf.nn.tanh(e_bn2)
            tf.summary.histogram('e_a2', e_a2)

            ### output layer
            self.q_eval = tf.layers.dense(e_a2, self.n_actions, activation=tf.nn.tanh, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')
            tf.summary.histogram('q_eval', self.q_eval)


        # ------------------ build target_net ------------------

        with tf.variable_scope('target_net'):
            # hidden layer 1
            t_z1 = tf.layers.dense(self.s_, 6, activation=None, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t1')
            tf.summary.histogram('t_z1', t_z1)

            t_bn1 = tf.layers.batch_normalization(t_z1, training=True)
            tf.summary.histogram('t_bn1', t_bn1)

            t_a1 = tf.nn.tanh(t_bn1)
            tf.summary.histogram('t_a1', t_a1)

            # hidden layer 2
            t_z2 = tf.layers.dense(t_a1, 6, activation=None, kernel_initializer=w_initializer,
                                   bias_initializer=b_initializer, name='t2')
            tf.summary.histogram('t_z2', t_z2)

            t_bn2 = tf.layers.batch_normalization(t_z2, training=True)
            tf.summary.histogram('t_bn2', t_bn2)

            t_a2 = tf.nn.tanh(t_bn2)
            tf.summary.histogram('t_a2', t_a2)

            ### output layer
            self.q_next = tf.layers.dense(t_a2, self.n_actions, activation=tf.nn.tanh, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t3')
            tf.summary.histogram('q_next', self.q_next)


        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
            # tf.summary.histogram('q_target', self.q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
            # tf.summary.histogram('q_eval', self.q_eval_wrt_a)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope('train'):
            with tf.control_dependencies(update_ops):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        self.reward.append(r)
        # transform a and r into 1D array
        transition = np.hstack((s, [a], [r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        _, cost, summary = self.sess.run(
            [self._train_op, self.loss, self.merged_summary],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)
        #self.writer.add_summary(summary, self.learn_step_counter)
        #self.writer.flush()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_reward(self, path, save):
        plt.plot(np.arange(len(self.reward)), self.reward)
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/reward.png')
        plt.show()

    def plot_cost(self, path, save):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/cost.png')
        plt.show()

    def save_model(self, path, count):
        self.saver.save(self.sess, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])