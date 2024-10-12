import torch as T
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
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

    def fit(self, env, n_episodes=1000, batch_size=32, t_max=200, penalty=-10):
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
                score = time
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
    def __init__(self, env, gamma=0.1, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def fit(self, n_episodes, t_max=2000, penalty=-10):
        for e in range(n_episodes):
            state = self.env.reset()
            score = 0
            for t in range(t_max):
                if np.random.rand() <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = penalty
                score += reward

                best_next_action = np.argmax(self.q_table[next_state])
                self.q_table[state, action] += self.learning_rate * (reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action])

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
    
    def fit(self, env, n_episodes=1000,t_max=2000):
        for i in range(n_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float)
            done = False
            Actions, States, Rewards = [], [], []
            t = 0
            while not done:
                t += 1
                if t>t_max:
                    break
                probs = self.model(state)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample().item()
                new_state, reward, done, _, __ = env.step(action)

                Actions.append(torch.tensor(action, dtype=torch.int))
                States.append(state)
                Rewards.append(reward)
                state = torch.tensor(new_state, dtype=torch.float)
            
            if self.score_type == 1:
                print(f'Episode: {i} Reward: {t}')
            else:
                print(f'Episode: {i} Reward: {sum(Rewards)}')

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
    
class PPO(nn.Module):
    def __init__(self, n_actions, input_dims, alpha=0.0003, gamma=0.99, policy_clip=0.2):
        super(PPO, self).__init__()
        self.gamma = gamma
        self.policy_clip = policy_clip

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = T.tensor(state, dtype=T.float).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)
        return dist, value

    def choose_action(self, state):
        dist, value = self.forward(state)
        dist = Categorical(dist)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value

    def learn(self, states, actions, log_probs, rewards, dones, values, batch_size, n_epochs):
        states = T.tensor(states, dtype=T.float).to(self.device)
        actions = T.tensor(actions, dtype=T.long).to(self.device)
        log_probs = T.tensor(log_probs).to(self.device)
        rewards = T.tensor(rewards).to(self.device)
        dones = T.tensor(dones).to(self.device)
        values = T.tensor(values).to(self.device)

        returns = []
        for t in range(len(rewards)):
            G = 0
            discount = 1
            for k in range(t, len(rewards)):
                G += rewards[k] * discount
                discount *= self.gamma
                if dones[k]:
                    break
            returns.append(G)
        returns = T.tensor(returns).to(self.device)

        advantages = returns - values.squeeze()

        for _ in range(n_epochs):
            for i in range(0, len(states), batch_size):
                states_batch = states[i:i + batch_size]
                actions_batch = actions[i:i + batch_size]
                old_log_probs_batch = log_probs[i:i + batch_size]
                advantages_batch = advantages[i:i + batch_size].detach()
                returns_batch = returns[i:i + batch_size].detach()

                dist, critic_value = self.forward(states_batch)
                dist = Categorical(dist)
                new_log_probs = dist.log_prob(actions_batch)

                prob_ratio = new_log_probs.exp() / old_log_probs_batch.exp()
                weighted_probs = advantages_batch * prob_ratio
                clipped_probs = advantages_batch * T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                actor_loss = -T.min(weighted_probs, clipped_probs).mean()

                critic_loss = (returns_batch - critic_value.squeeze()) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def train(self, env, n_games, N, batch_size, n_epochs, alpha):
        best_score = env.reward_range[0]
        score_history = []
        learn_iters = 0
        avg_score = 0
        n_steps = 0

        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0
            states = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            dones = []

            while not done:
                action, prob, val = self.choose_action(observation)
                observation_, reward, done, _ = env.step(action)

                n_steps += 1
                score += reward

                states.append(observation)
                actions.append(action)
                log_probs.append(prob)
                values.append(val)
                rewards.append(reward)
                dones.append(done)

                observation = observation_

                if n_steps % N == 0:
                    self.learn(states, actions, log_probs, rewards, dones, values, batch_size, n_epochs)
                    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
                    learn_iters += 1

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            print(f'Episode {i}, Score {score:.1f}, Avg Score {avg_score:.1f}, Time Steps {n_steps}, Learning Steps {learn_iters}')

        x = [i + 1 for i in range(len(score_history))]
        return x, score_history

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class A3C(nn.Module):
    def __init__(self,env, input_dims, n_actions, gamma=0.99, lr=1e-3, n_games=5000, t_max=5):
        super(A3C, self).__init__()

        
        self.gamma = gamma
        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)
        self.env=env
        self.n_games = n_games
        self.t_max = t_max

        
        self.global_actor_critic = self
        self.global_actor_critic.share_memory()

        
        self.optimizer = SharedAdam(self.global_actor_critic.parameters(), lr=lr)
        
        
        self.global_ep_idx = mp.Value('i', 0)

        
        self.rewards = []
        self.actions = []
        self.states = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))
        pi = self.pi(pi1)
        v = self.v(v1)
        return pi, v

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)
        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        return T.tensor(batch_return, dtype=T.float)

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)
        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()

        critic_loss = (returns - values) ** 2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()
        return total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, _ = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]
        return action

    def worker(self, name):
        local_actor_critic = A3C(self.pi1.in_features, self.pi.out_features, self.gamma, n_games=self.n_games, t_max=self.t_max)
        while self.global_ep_idx.value < self.n_games:
            done = False
            observation = self.env.reset()
            score = 0
            local_actor_critic.clear_memory()
            t_step = 1
            while not done:
                action = local_actor_critic.choose_action(observation)
                observation_, reward, done, _ = self.env.step(action)
                score += reward
                local_actor_critic.remember(observation, action, reward)

                if t_step % self.t_max == 0 or done:
                    loss = local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    local_actor_critic.clear_memory()

                t_step += 1
                observation = observation_

            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1

            print(f'Worker {name}, Episode {self.global_ep_idx.value}, Score: {score:.1f}')

    def fit(self, n_workers=4):
        workers = []
        for i in range(n_workers):
            worker_process = mp.Process(target=self.worker, args=(i,))
            workers.append(worker_process)

        for worker_process in workers:
            worker_process.start()

        for worker_process in workers:
            worker_process.join()