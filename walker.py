import pickle
import scipy.signal
from dm_control import suite, viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm


class policy_net(nn.Module):
    def __init__(self, xdim, udim, hdim=128, fixed_var=False):
        super().__init__()

        self.xdim = xdim
        self.udim = udim
        self.fixed_var = fixed_var

        self.fc1 = nn.Linear(xdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, hdim)
        self.fc_mu = nn.Linear(hdim, udim)

        if self.fixed_var:
            self.fc_log_std = None
        else:
            self.fc_log_std = nn.Linear(hdim, udim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mu = self.fc_mu(x)

        if self.fixed_var:
            log_std = torch.log(torch.ones_like(mu) * 0.1)
        else:
            log_std = self.fc_log_std(x)
        std = torch.exp(log_std)

        return mu, std


class value_net(nn.Module):
    def __init__(self, xdim, hdim=128):
        super().__init__()

        self.fc1 = nn.Linear(xdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, hdim)
        self.fc_value = nn.Linear(hdim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.fc_value(x)
        return value


class PPO:
    def __init__(self, xdim, udim, hdim=32, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lambd=0.97, K_epochs=6, eps_clip=0.2):

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # actor and critic are on GPU
        self.actor = policy_net(xdim, udim, hdim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)

        self.critic = value_net(xdim, hdim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lambd = lambd
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

    # input: numpy 0d state, CPU
    # output: numpy 0d action, CPU
    def select_action(self, state):
        # add a dimension, to 1 x xdim, tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # forward pass through policy network
        mu, std = self.actor(state)

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()

        return action.detach().cpu().numpy()[0]

    def update(self, buffer):

        def compute_advantage(x, discount):
            # Convert tensor to numpy, move to CPU
            x = x.detach().cpu().numpy()
            result = scipy.signal.lfilter(
                [1], [1, float(-discount)], x[::-1], axis=0)[::-1]
            # Convert back to tensor
            return torch.tensor(result.copy(), dtype=torch.float).to(self.device)

        s = torch.tensor(
            np.array(buffer['x']), dtype=torch.float).to(self.device)
        a = torch.tensor(
            np.array(buffer['u']), dtype=torch.float).to(self.device)
        r = torch.tensor(
            np.array(buffer['r']), dtype=torch.float).view(-1, 1).to(self.device)
        next_s = torch.tensor(
            np.array(buffer['next_x']), dtype=torch.float).to(self.device)
        done = torch.tensor(
            np.array(buffer['done']), dtype=torch.float).view(-1, 1).to(self.device)

        assert s.shape[1] == xdim
        assert a.shape[1] == udim
        assert next_s.shape[1] == xdim
        assert s.shape[0] == a.shape[0] == r.shape[0] == next_s.shape[0] == done.shape[0]

        TD = r + self.gamma * self.critic(next_s) * (1 - done)
        delta = TD - self.critic(s)
        # GAE advantage
        advantage = compute_advantage(
            delta, self.gamma * self.lambd)  # Compute advantage on CPU

        assert advantage.shape == done.shape

        # TODO: add advantage normalization

        mu, std = self.actor(s)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(a)

        for _ in range(self.K_epochs):
            mu, std = self.actor(s)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(a)
            ratio = torch.exp(log_probs - old_log_probs)
            # TODO: PPO penalty
            option1 = ratio * advantage
            option2 = torch.clamp(ratio, 1 - self.eps_clip,
                                  1 + self.eps_clip) * advantage
            actor_loss = -torch.min(option1, option2).mean()
            critic_loss = F.mse_loss(self.critic(s), TD.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()


# all variables should be numpy 0d
def rollout(env, agent, T=1000):
    episode_return = 0
    episode_traj = {'x': [], 'u': [],
                    'next_x': [], 'r': [], 'done': []}

    t = env.reset()
    # state
    x = t.observation
    x = np.array(x['orientations'].tolist() +
                 [x['height']]+x['velocity'].tolist())
    # done
    done = False

    for _ in range(T):

        u = agent.select_action(x)
        r = env.step(u)

        # state
        next_x = r.observation
        next_x = np.array(next_x['orientations'].tolist() +
                          [next_x['height']]+next_x['velocity'].tolist())
        # reward
        reward = r.reward
        # done
        done = r.last()

        assert x.shape == (xdim,)
        assert u.shape == (udim,)
        assert next_x.shape == (xdim,)
        assert reward.shape == ()
        assert done == True or done == False

        episode_traj['x'].append(x)
        episode_traj['u'].append(u)
        episode_traj['next_x'].append(next_x)
        episode_traj['r'].append(reward)
        episode_traj['done'].append(done)

        episode_return += reward

        x = next_x

        if done:
            break

    return episode_traj, episode_return


def train(env, agent, num_episodes):
    return_list = []

    for i in range(num_episodes):
        # the traj and return of each episode
        episode_traj, episode_return = rollout(env, agent)

        # print(f'Episode {i}: return {episode_return}')

        # record the return of each episode
        return_list.append(episode_return)

        # update the agent with the episode's trajectory
        agent.update(episode_traj)

        if (i+1) % 1 == 0:
            print(f'Episode {i+1}: return {episode_return}')

        if (i + 1) % 500 == 0:
            torch.save(agent.actor.state_dict(), f'ppo_actor_{i + 1}.pth')
            torch.save(agent.critic.state_dict(), f'ppo_critic_{i + 1}.pth')
            with open(f'return_list_{i + 1}.pkl', 'wb') as f:
                pickle.dump(return_list, f)
            print(f'Model and return list saved at episode {i + 1}')

    # return return_list
    return agent


def load_model(agent, task_name, window_size=100):
    """根据任务名称加载模型和绘制回报图表"""

    def moving_average(data, window_size):
        """计算滑动窗口平均"""
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_size:] -
                  cumsum_vec[:-window_size]) / window_size
        return ma_vec

    # 根据任务名称构建文件路径
    actor_path = f'./{task_name}_model/ppo_actor_5000.pth'
    critic_path = f'./{task_name}_model/ppo_critic_5000.pth'
    return_list_path = f'./{task_name}_model/return_list_5000.pkl'

    # 加载模型
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))

    # 加载回报列表并绘制图表
    with open(return_list_path, 'rb') as f:
        loaded_return_list = pickle.load(f)

    # 绘制原始回报图表
    plt.figure(figsize=(10, 5))
    plt.plot(loaded_return_list, label='Episode Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'{task_name.capitalize()} Task - Return per Episode Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{task_name}_return_plot.png')
    plt.show()

    # 计算并绘制滑动平均回报图表
    smoothed_returns = moving_average(loaded_return_list, window_size)
    plt.figure(figsize=(12, 6))
    plt.plot(loaded_return_list, label='Original Returns', alpha=0.5)
    plt.plot(np.arange(window_size-1, len(loaded_return_list)),
             smoothed_returns, label='Smoothed Returns', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(
        f'{task_name.capitalize()} Task - Return per Episode with Moving Average')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{task_name}_smoothed_return_plot.png')
    plt.show()

    return agent


if __name__ == '__main__':
    r0 = np.random.RandomState(42)
    env = suite.load('walker', 'walk',
                     task_kwargs={'random': r0})
    U = env.action_spec()
    udim = U.shape[0]
    X = env.observation_spec()
    xdim = 14+1+9

    agent = PPO(xdim, udim)

    agent = train(env, agent, 5000)
    # task_name = 'walk'  # 或 'stand'
    # agent = load_model(agent, task_name)

    '''
    def u(dt):
        return np.random.uniform(low=U.minimum,
                                high=U.maximum,
                                size=U.shape)
    viewer.launch(env, policy=u)
    '''

    def u(timestep):
        x = timestep.observation
        x = np.array(x['orientations'].tolist() +
                     [x['height']] +
                     x['velocity'].tolist())

        return agent.select_action(x)

    viewer.launch(env, policy=u)
