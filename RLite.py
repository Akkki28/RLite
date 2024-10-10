import random
import numpy as np
from collections import deque
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
class DQN:
    def __init__(self, state_size, action_size,gamma = 0.95,epsilon = 1.0,epsilon_decay = 0.995,epsilon_min = 0.01,learning_rate = 0.001,loss='mse',n_layers=24):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min 
        self.learning_rate= learning_rate
        self.loss = loss
        self.n_layers=n_layers
        self.model = self._build_model() 
    
    def _build_model(self):
        
        model = Sequential()
        model.add(Dense(self.n_layers, input_dim=self.state_size, activation='relu')) 
        model.add(Dense(self.n_layers, activation='relu')) 
        model.add(Dense(self.action_size, activation='linear')) 
        model.compile(loss=self.loss,
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def act(self, state):
        if np.random.rand() <= self.epsilon: 
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) 
        return np.argmax(act_values[0]) 

    def replay(self, batch_size): 
        minibatch = random.sample(self.memory, batch_size) 
        for state, action, reward, next_state, done in minibatch: 
            target = reward 
            if not done: 
                target = (reward + self.gamma * 
                          np.amax(self.model.predict(next_state)[0])) 
            target_f = self.model.predict(state) 
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    
    def fit(self,env,n_episodes=1000,batch_size=32,t_max=200,score_type=1,penalty=-10):
        output_dir = 'model_output/weights/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        done = False
        for e in range(n_episodes): 
            state = env.reset() 
            state=state[0]
            state = np.reshape(state, [1,self.state_size])
            score = 0
            for time in range(t_max):  

                action = self.act(state) 
                next_state, reward, done, _ ,_= env.step(action) 
                reward = reward if not done else penalty 
                if score_type==1:
                    score=time
                else:
                    score+=reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done) 
                state = next_state 
                if done: 
                    print("episode: {}/{}, score: {}" 
                        .format(e, n_episodes, score))
                    break 
            if len(self.memory) > batch_size:
                self.replay(batch_size) 
            if e % 100 == 0:
                self.save(output_dir + "weights_" + '{:04d}'.format(e) + ".weights.h5")      


class Q:
    def __init__(self,env,gamma = 0.1,epsilon = 1.0,epsilon_decay = 0.995,epsilon_min = 0.01,learning_rate = 0.001):
        self.env=env
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min 
        self.learning_rate= learning_rate
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def fit(self,n_episodes,t_max=2000,score_type=1,penalty=-10):
        
        for e in range(n_episodes):
            state = self.env.reset()
            score = 0
            for t in range(t_max):
                if np.random.rand() <= self.epsilon:
                    action = self.env.action_space.sample()  
                else:
                    action = np.argmax(self.q_table[state])
                next_state, reward, done, _ = self.env.step(action)
                score+=reward
                if score_type==1:
                    if done :
                        reward=penalty
                    score = t
                else :
                    score += reward
                
                best_next_action = np.argmax(self.q_table[next_state])
                self.q_table[state, action] += 1*(reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action])
        
                state = next_state
                if done:
                    break
            if e % 100 == 0:
                print(f"Episode {e}/{n_episodes}, Total Reward: {score}")

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def show_q_table(self):
        print(self.q_table)


class REINFORCE:
    def __init__(self,state_size, action_size,lr=0.05,gamma=0.99,n_layers=64):
        self.state_size=state_size
        self.action_size=action_size
        self.lr=lr
        self.gamma=gamma
        self.n_layers=n_layers
        self.model = self._build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

    def _build_model(self):
        model = keras.Sequential([
        keras.layers.Dense(self.n_layers, activation='relu', input_shape=(self.state_size,)),
        keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model
    
    def fit(self,env,n_episodes=1000):
        for i in range(n_episodes):
            state = env.reset()
            done = False
            Actions, States, Rewards = [], [], []

            while not done:
                state = np.expand_dims(state, axis=0)
                probs = self.model(state)
                action = np.random.choice(self.action_size, p=np.squeeze(probs))

                new_state, rew, done, _ = env.step(action)
                
                Actions.append(action)
                States.append(state)
                Rewards.append(rew)

                state=new_state

            DiscountedReturns = []
            G = 0.0
            for t in reversed(range(len(Rewards))):
                G = Rewards[t] + self.gamma * G
                DiscountedReturns.insert(0, G) 
            DiscountedReturns = np.array(DiscountedReturns)
            DiscountedReturns = (DiscountedReturns - np.mean(DiscountedReturns)) / (np.std(DiscountedReturns) + 1e-8)
            with tf.GradientTape() as tape:
                total_loss = 0
                for State, Action, G in zip(States, Actions, DiscountedReturns):
                    State = np.expand_dims(State, axis=0)
                    probs = self.model(State)
                    log_prob = tf.math.log(probs[0, Action])
                    total_loss += -log_prob * G

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


            




        
