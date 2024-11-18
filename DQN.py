import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import gym

class ReplayBuffer:
    """经验回放"""
    def __init__(self,batch_size=64):
        self.data = []
        self.batch_size = batch_size
    
    def add_data(self,state,action,reward,next_state,done):
        self.data.append((state,action,reward,next_state,done))
    
    def sample(self):
        sample_data = random.sample(self.data,self.batch_size)
        state, action, reward, next_state, done = zip(*sample_data)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def length(self):
        return len(self.data)

class model(nn.Module):
    """定义Q神经网络"""
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(model,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
            )
        
    def forward(self,x):
        return self.net(x)

class DQN():
    """DQN智能体"""
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device):
        self.action_dim = action_dim
        self.net = model(state_dim,hidden_dim,action_dim).to(device)
        self.load()
        self.target_net = model(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=learning_rate)
        self.gamma = gamma # 奖励折扣因子
        self.epsilon = epsilon #贪婪行为策略的参数
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 # 记录更新次数
        self.device = device
        self.loss = nn.MSELoss(reduction='none')
        
    def take_action(self,state):
        if random.random() < self.epsilon:
            action = random.randint(0,self.action_dim-1)
        else:
            state = torch.tensor(state,dtype=torch.float).to(self.device)
            action = self.net(state).argmax().item()
        return action
    
    def update(self,states,actions,rewards,next_states,dones):
        states = torch.tensor(states,dtype=torch.float).to(device=self.device)
        actions = torch.tensor(actions,dtype=torch.int64).reshape(-1,1).to(device=self.device) # 转换成2维多行一列，是为了之后提取对应动作的价值方便
        rewards = torch.tensor(rewards,dtype=torch.float).to(device=self.device)
        next_states = torch.tensor(next_states,dtype=torch.float).to(device=self.device)
        dones = torch.tensor(dones,dtype=torch.float).to(device=self.device)
        
        q_values = self.net(states).gather(-1,actions) # gather第二参数索引必须为tensor类型
        max_q_values = self.target_net(next_states).max(dim=-1)[0] # 指定维度后，返回值：最大值 + 索引
        q_target = rewards + self.gamma * max_q_values * (1-dones)
        q_target = q_target.unsqueeze(dim=-1)
        l = self.loss(q_values,q_target)
        self.optimizer.zero_grad()
        l.mean().backward()
        self.optimizer.step()
        
        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.count += 1
    
    def save(self):
        torch.save(self.net.state_dict(),'DQN_CartPole.pth')
    
    def load(self):
        try:
            self.net.load_state_dict(torch.load('DQN_CartPole.pth'))
        except:
            pass

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    lr = 2e-3
    num_episodes = 20
    gamma = 0.9
    hidden_dim = 128
    epsilon = 0.05
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make('CartPole-v1',render_mode='human')
    # env = gym.make('CartPole-v1')
    replay_buffer = ReplayBuffer(batch_size)
    state_dim = env.observation_space.shape[0]
    print(state_dim)
    action_dim = env.action_space.n
    agent = DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device)
    return_list = []
    
    for i in range(10):
        for i_episode in range(num_episodes):
            episode_return = 0
            state, info = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _, __ = env.step(action)
                replay_buffer.add_data(state,action,reward,next_state,done)
                state = next_state
                episode_return += reward
                if replay_buffer.length() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample()
                    agent.update(b_s, b_a, b_r, b_ns, b_d)
            print(episode_return)
            return_list.append(episode_return)
    agent.save()