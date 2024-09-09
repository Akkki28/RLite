import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
class DQN(nn.Module):
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001, n_layers=24):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, self.n_layers),
            nn.ReLU(),
            nn.Linear(self.n_layers, self.n_layers),
            nn.ReLU(),
            nn.Linear(self.n_layers, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            action = torch.LongTensor([action])
            
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            current_q_value = self.model(state)[0][action]
            loss = F.mse_loss(current_q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def fit(self, env, n_episodes=1000, batch_size=32, t_max=200, score_type=1, penalty=-10):
        output_dir = 'model_output/weights/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for e in range(n_episodes):
            state = env.reset()
            state = state[0]
            state = np.reshape(state, [1, self.state_size])
            score = 0
            for time in range(t_max):
                action = self.act(state)
                next_state, reward, done, _, _ = env.step(action)
                reward = reward if not done else penalty
                if score_type == 1:
                    score = time
                else:
                    score += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"episode: {e}/{n_episodes}, score: {score}")
                    break
            if len(self.memory) > batch_size:
                self.replay(batch_size)
            if e % 100 == 0:
                self.save(output_dir + f"weights_{e:04d}.pth")
      


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


class REINFORCE(nn.Module):
    def __init__(self, state_size, action_size, lr=0.005, gamma=0.9999, n_layers=128):
        super(REINFORCE, self).__init__() 
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.n_layers = n_layers
        self.model = self._build_model() 
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) 

    def _build_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, self.n_layers),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_layers, self.action_size),
            torch.nn.Softmax(dim=-1)
        )
        return model
    
    def fit(self, env, n_episodes=1000):
        for i in range(n_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float)
            done = False
            Actions, States, Rewards = [], [], []
            t = 0
            while not done:
                t += 1
                if t>200:
                    break
                probs = self.model(state)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample().item()
                new_state, reward, done, _, __ = env.step(action)

                Actions.append(torch.tensor(action, dtype=torch.int))
                States.append(state)
                Rewards.append(reward)
                state = torch.tensor(new_state, dtype=torch.float)
            
            print(f'Episode: {i} Reward: {t}')

            DiscountedReturns = []
            for t in range(len(Rewards)):
                G = 0.0
                for k, r in enumerate(Rewards[t:]):
                    G += (self.gamma**k) * r
                DiscountedReturns.append(G)

            for State, Action, G in zip(States, Actions, DiscountedReturns):
                probs = self.model(State)
                dist = torch.distributions.Categorical(probs=probs)
                log_prob = dist.log_prob(Action)
                
                loss = -log_prob * G
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def save(self, name):
        torch.save(self.model.state_dict(), name)
    


