# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.compat.v1 as tf  
tf.compat.v1.disable_eager_execution()
import os
import random
import keras.backend as K
from keras.layers import Input, Dense,concatenate
from keras.models import Model
from keras.optimizers import Adam
from UDN_env import UDNEnv
from matplotlib import pyplot as plt
from collections import deque



class DDPG(UDNEnv):
    def __init__(self):
        super(DDPG, self).__init__()

        self.sess = K.get_session() 
        # import the reinforcement learning environment
        self.env = UDNEnv()
        # configure the dimension of state and action 
        self.s_dim=self.env.s_dim 
        self.a_dim =self.env.a_dim
        self.bound=1
        # update rate for target model.
        self.TAU = 0.01
        # experience replay.
        self.memory_buffer = deque(maxlen=4000)
        # discount rate for Q value.
        self.gamma = 0.95
        # epsilon of action selection
        self.epsilon = 1.0
        # discount rate for epsilon.
        self.epsilon_decay = 0.995
        # min epsilon of ε-greedy.
        self.epsilon_min = 0.01

        # actor learning rate
        self.a_lr = 0.0001
        # critic learining rate
        self.c_lr = 0.001

        # DDPG model
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # target model
        self.target_actor = self._build_actor()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic = self._build_critic()
        self.target_critic.set_weights(self.critic.get_weights())

        # gradient function
        self.get_critic_grad = self.critic_gradient()
        self.actor_optimizer()

        if os.path.exists('model/ddpg_actor.h5') and os.path.exists('model/ddpg_critic.h5'):
            self.actor.load_weights('model/ddpg_actor.h5')
            self.critic.load_weights('model/ddpg_critic.h5')
    def _build_actor(self):
        """Actor model
        """
        inputs = Input(shape=(3,), name='state_input')  #输入状态
        x = Dense(40, activation='relu')(inputs)
        x = Dense(40, activation='relu')(x)
        output= Dense(2, activation='tanh')(x)     # 预测动作

        model = Model(inputs=inputs, outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.a_lr))

        return model
    def _build_critic(self):
        """Critic model
        """
        sinput = Input(shape=(3,), name='state_input')  #输入1：状态N*3维
        ainput = Input(shape=(2,), name='action_input')  #输入2：动作N*2维
        s = Dense(40, activation='relu')(sinput)
        a = Dense(40, activation='relu')(ainput)
        x = concatenate([s, a])
        x = Dense(40, activation='relu')(x)
        output = Dense(1, activation='tanh')(x)     #预测Q值：1维

        model = Model(inputs=[sinput, ainput], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.c_lr))

        return model
    
    
    def actor_optimizer(self):
        """actor_optimizer.
        Returns:
            function, opt function for actor.
        """
        self.ainput = self.actor.input
        aoutput = self.actor.output
        trainable_weights = self.actor.trainable_weights
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, 2)) #动作梯度

        # tf.gradients will calculate dy/dx with a initial gradients for y
        # action_gradient is dq / da, so this is dq/da * da/dparams
        params_grad = tf.gradients(aoutput, trainable_weights, -self.action_gradient)
        grads = zip(params_grad, trainable_weights)
        self.opt = tf.train.AdamOptimizer(self.a_lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    
    def critic_gradient(self):
        """get critic gradient function.

        Returns:
            function, gradient function for critic.
        """
        cinput = self.critic.input
        coutput = self.critic.output

        # compute the gradient of the action with q value, dq/da.
        action_grads = K.gradients(coutput, cinput[1])

        return K.function([cinput[0], cinput[1]], action_grads)
    
    
    def OU(self, x, mu=0, theta=0.15, sigma=0.2):
       """Ornstein-Uhlenbeck process.
       formula：ou = θ * (μ - x) + σ * w

       Arguments:
           x: action value.
           mu: μ, mean fo values.
           theta: θ, rate the variable reverts towards to the mean. 
           sigma：σ, degree of volatility of the process.

       Returns:
           OU value
       """
       return theta * (mu - x) + sigma * np.random.randn(1)
   
    
   
    def get_action(self, s):    #下一动作
        """get actor action with ou noise.

        Arguments:
            s: state value.
        """
        action = self.actor.predict(s)  #预测连续动作

        # add randomness to action selection for exploration
        noise = max(self.epsilon, 0) * self.OU(action)  #add  noise
        action = np.clip(action + noise, -self.bound, self.bound)
        
        action=np.abs(action)    # 动作状态变量取正

        return action

    def remember(self, state, action, reward, next_state, done):   #缓存记忆每轮训练参数
        """add data to experience replay.

        Arguments:
            state: observation.
            action: action.
            reward: reward.
            next_state: next_observation.
            done: if game done.
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)
    
    def update_epsilon(self):  #贪婪算法
        """update epsilon.
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            
    def process_batch(self, batch):  #处理采样batch
        """process batch data.

        Arguments:
            batch: batch size.

        Returns:
            states: states.
            actions: actions.
            Q: Q_value.
        """
        Q = []  #记录Q值
        data = random.sample(self.memory_buffer, batch)  # 历史经验数据随机采样，采样大小batch
        states = np.array([d[0] for d in data]).reshape(batch*self.s_dim[0],self.s_dim[1]) # (128*12,12,3)
        actions = np.array([d[1] for d in data]).reshape(batch*self.a_dim[0],self.a_dim[1]) # (128*12,12,2)
        next_states = np.array([d[3] for d in data]).reshape(batch*self.s_dim[0],self.s_dim[1]) #下一状态

        # Q_target
        next_actions = self.target_actor.predict(next_states)
        q = self.target_critic.predict([next_states, next_actions])

        # update Q value
        for i, (_,_,reward,_,done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * q[i][0]
            Q.append(target)

        return states, actions, Q

    def update_model(self, states, actions, Q):
        """update ddpg model.

        Arguments:
            states: states.
            actions: actions.
            Q: Q_value.

        Returns:
            loss: critic loss.
        """
#        loss = self.critic.train_on_batch([X1, X2], y)
        batch_size=int(states.shape[0]/self.env.num_ue)
        loss=[]
        for i in range(0,self.env.num_ue):
           train_loss = self.critic.fit([states[batch_size*i:batch_size*(i+1),:], actions[batch_size*i:batch_size*(i+1),:]], Q, verbose=0)
           loss.append( train_loss.history['loss']) 
        loss = np.mean(loss)

        action_s= self.actor.predict(states)
        a_grads = np.array(self.get_critic_grad([states, action_s]))[0]
        
        self.sess.run(self.opt, feed_dict={
            self.ainput: states,
            self.action_gradient: a_grads
        })

        return loss

    def update_target_model(self):
        """soft update target model.
        formula：θ​​t ← τ * θ + (1−τ) * θt, τ << 1. 
        """
        critic_weights = self.critic.get_weights()
        actor_weights = self.actor.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        actor_target_weights = self.target_actor.get_weights()

        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]

        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]

        self.target_critic.set_weights(critic_target_weights)
        self.target_actor.set_weights(actor_target_weights)

    def train(self,episode,batch):
        """training model.
        Arguments:
            episode: ganme episode.
            batch： batch size of episode.

        Returns:
            history: training history.
        """
        history = {'episode': [], 'Episode_reward': [], 'Loss': [],'time_cost':[],'energy_cost':[]}
        for i in range(episode):
          observation = self.env.reset() #更新观察变量：状态task_size,delay_max,ue_uplink_rate)
          reward_sum = 0
          losses = []
          delay=[]
          e=[]
          for j in range(100):
               action = self.get_action(observation)
               total_reward,data_rate_uplink,done,total_delay,total_e= self.env.step(action)
               reward_sum +=total_reward 
               delay.append(total_delay)
               e.append(total_e)
               next_observation=self.env.reset()
               # next_observation[:,2]=data_rate_uplink     #更新UE观察状态变量
               self.remember(observation,action,total_reward,next_observation,done)
               if len(self.memory_buffer) > batch:
                    states,actions, Q = self.process_batch(batch)
                    # update DDPG model
                    loss = self.update_model(states,actions, Q)
                    # update target model
                    self.update_target_model()
                    # reduce epsilon pure batch.
                    self.update_epsilon()
                    losses.append(loss)
               
          time_delay=np.mean(delay) 
          energy_cost=np.mean(e)
          loss = np.mean(losses)
          history['episode'].append(i)
          history['Episode_reward'].append(round(reward_sum/(episode*100),2))
          history['Loss'].append(loss)
          history['time_cost'].append(time_delay)  #平均时延
          history['energy_cost'].append(energy_cost) #平均能耗
          print('Episode: {}/{} | reward: {} | loss: {:.3f} | time_cost: {:.3f} | energy_cost: {:.3f} |'.format(i, episode, reward_sum, loss, time_delay, energy_cost))

        # self.actor.save_weights('model/ddpg_actor.h5')
        # self.critic.save_weights('model/ddpg_critic.h5')
        plt.plot(history['Episode_reward'],'k--')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
        return history
    
    def play(self):
        """play game with model.
        """
        print('play...')
        observation = self.env.reset()

        reward_sum = 0
        random_episodes = 0

        while random_episodes < 100:
            x = observation
            action = self.actor.predict(x)[0]
            total_reward,data_rate_uplink,done,total_delay,total_e = self.env.step(action)
            reward_sum += total_reward

            print("Reward for this episode was: {}".format(reward_sum))
            random_episodes += 1
            reward_sum = 0
            observation = self.env.reset()       

if __name__ == '__main__':
    model = DDPG()
    
    
    
    
    history = model.train(20, 64)
    # model.save_history(history, 'ddpg.csv')
    # model.play()
        
 










