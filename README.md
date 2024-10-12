# RLite
a Python library that streamlines the implementation of reinforcement learning algorithms supporting both finite and infinite state spaces.

## DQN (Deep Q-Network)

The DQN class implements the DQN algorithm. DQN is a combination of Q-learning and deep learning techniques, using a neural network to approximate the Q-value function. This allows it to handle complex environments with large state spaces effectively. DQN improves upon traditional Q-learning by utilizing experience replay and target networks to stabilize training.

### Key Features:
- **Experience Replay**: Stores past experiences to break the correlation between consecutive samples.
- **Target Network**: A separate network used to compute target Q-values, updated periodically to improve stability.

## Q-learning

The QLearning class implements the Q-learning algorithm. Q-learning is a value-based method that aims to learn the optimal action-value function (Q-function) for a given policy. It updates the Q-values iteratively based on the Bellman equation, allowing the agent to learn from its actions without requiring a model of the environment.

### Key Features:
- **Off-Policy Learning**: Learns the value of the optimal policy independently from the policy being followed.
- **Exploration vs. Exploitation**: Balances between exploring new actions and exploiting known rewards.

## REINFORCE

The REINFORCE class implements the REINFORCE algorithm. REINFORCE is a policy gradient method that directly optimizes the policy by maximizing the expected return. It uses Monte Carlo sampling to estimate the gradients of the policy, enabling the agent to learn from complete episodes of experience.

### Key Features:
- **Policy-Based**: Focuses on learning the policy directly rather than estimating the value function.
- **High Variance**: The gradients can have high variance, which may lead to unstable training.

## PPO (Proximal Policy Optimization)

The PPO class implements the PPO algorithm. PPO is an advanced policy gradient method that aims to improve training stability and efficiency. It uses a clipped surrogate objective to limit the changes to the policy during each update, ensuring that the new policy does not deviate too much from the old one.

### Key Features:
- **Clipped Objective**: Prevents large updates that could degrade performance.
- **Adaptive Learning Rate**: Adjusts the learning rate based on the performance of the agent.

## A3C (Asynchronous Actor-Critic)

The A3C class implements the A3C algorithm. A3C is a hybrid algorithm that combines the advantages of both value-based and policy-based methods. It uses multiple parallel agents to explore the environment and share gradients, leading to faster and more stable training.

### Key Features:
- **Asynchronous Training**: Multiple agents interact with the environment simultaneously, sharing their experiences.
- **Actor-Critic Architecture**: Uses both a policy network (actor) and a value network (critic) to improve learning efficiency.

## Usage

To use these models, create an instance of the class you want to use, and call the `fit` method with your suitable environment.If the evironment is not suitable according to the given implementation change the code with the requiremnts of the given environment

## Requirements
### Installing Dependencies
```
pip install -r requirements.txt
```
