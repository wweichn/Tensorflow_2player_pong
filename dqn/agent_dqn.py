import tensorflow as tf
import random as random
from multinet import *
import numpy as np
from history import *
from replay_memory import *


INITIAL_EPSILONS = [0.4,0.3,0.3]
FINAL_EPSILONS = [0.01,0.01,0.05]
REPLACE_ITER = 500
EPSILONS = 3
Actions = 4

class Agent_dqn(object):
    def __init__(self,sess,config,name):

        self.sess = sess
        self.name =  str(name)
        self.min_reward = -1
        self.max_reward = 1
        self.learn_start = config.learn_start
        self.train_frequency = config.train_frequency
        self.target_q_update = config.target_q_update
        self.update_count = 0
        self.step = 0
        self.total_q = 0
        self.total_loss = 0
        self.memory = ReplayMemory(config,'../model')
        self.history_length = config.history_length
        self.replace_counter = config.replace_counter
        self.discount = config.GAMMA
        self.network = MultilayerNetwork(self.name,input_width = config.screen_width, input_height = config.screen_height, nimages = 4)
        self.update_target_q_network()
        self.build_summary()
        self.ep_start = 1
        self.ep_end = 0.1
        self.ep_end_t = config.ep_end_t
        self.saver = tf.train.Saver(max_to_keep = 30)
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

    def select_action(self,input_s, test_ep = None):
        ep = test_ep or (self.ep_end +
                         max(0, (self.ep_start - self.ep_end) *
                             (self.ep_end_t - max(0, self.step - self.learn_start)) / self.ep_end_t))

        if random.random() <= ep:
            action_index = random.randrange(Actions)
        else:
            print('net')
            action_index = self.network.q_action.eval({self.network.input_image: [input_s]})[0]
  #          action_index = self.sess.run(self.network.readout_a, feed_dict = {self.input_image:input_s})

        return action_index


    def perceive(self, screen, reward, action, terminal):

        reward = max(self.min_reward, min(self.max_reward, reward))
 #      reward = reward
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update == 0:
                self.update_target_q_network()

    def q_learning_mini_batch(self):

        if self.memory.count < self.history_length:
            return

        else:
            s, action, reward, s_next, terminal = self.memory.sample()

        pred_action = self.network.q_action.eval({self.network.input_image:s_next})

        #print(self.sess.run(tf.shape(pred_action)))

        q_t_next_state = self.network.q_real_.eval({
            self.network.input_image:s_next, self.network.input_a:pred_action
        })

        target_q = (1 - terminal) * self.discount * q_t_next_state + reward

        loss = self.learn(state_batch = s, value_batch = target_q, action_batch = action)
        self.update_count += 1
        self.total_loss += loss
        self.total_q += target_q.mean()

    def save_model(self):
        str_modelpath = "./model/" + self.name + "/model.ckpt"
        self.saver.save(self.sess, str_modelpath, self.step)

    def save_model_epoch(self,epoch):
        str_modelpath = "./model_/" + self.name + "/model.ckpt"
        self.saver.save(self.sess, str_modelpath,epoch )

    def load_model(self):
        checkpoint_dir = "./model/" + self.name + "/"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print( "OLD VARS")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("NEW VARS")

    def learn(self, state_batch,value_batch, action_batch):
        loss,_ = self.sess.run([self.network.cost, self.network.train_step],feed_dict={self.network.input_image: state_batch, self.network.input_y: value_batch, self.network.input_a: action_batch})
        return loss

    def build_summary(self):
        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q',
                                   'episode.max_reward', 'episode.min_reward', 'episode.avg_reward','episode.num_of_game']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name = tag)
                self.summary_ops[tag] = tf.summary.scalar("%s/%s" % (self.name, tag), self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name = tag)
                self.summary_ops[tag] = tf.summary.histogram("%s/%s" %(self.name, tag), self.summary_placeholders[tag])


    def inject_summary(self, tag_dict):

        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]:value for tag, value in tag_dict.items()})\

        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step)

    def update_target_q_network(self):
        self.ne_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name + "/source")
        self.nt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name + "/target")
        self.assign_op = []

        for w, t_w in zip(self.ne_params, self.nt_params):
            self.assign_op.append(t_w.assign(w))





