import pickle
import scipy.signal
from dm_control import suite, viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, xdim, udim, hdim=64):
        super().__init__()

        self.xdim = xdim
        self.udim = udim

        self.fc1 = nn.Linear(xdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, hdim)
        self.fc_mu = nn.Linear(hdim, udim)
        self.fc_log_std = nn.Linear(hdim, udim)

    def forward(self, x):
        # Trick: tanh activation
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))

        mu = self.fc_mu(x)

        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)

        return mu, std


class ValueNet(nn.Module):
    def __init__(self, xdim, hdim=64):
        super().__init__()

        self.fc1 = nn.Linear(xdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, hdim)
        self.fc_value = nn.Linear(hdim, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))

        value = self.fc_value(x)
        return value


class ReplayBuffer:
    def __init__(self, batch_size, xdim, udim):
        self.s = np.zeros((batch_size, xdim))
        self.a = np.zeros((batch_size, udim))
        self.a_logp = np.zeros((batch_size, udim))
        self.r = np.zeros((batch_size, 1))
        self.next_s = np.zeros((batch_size, xdim))
        self.done = np.zeros((batch_size, 1))
        self.count = 0
        self.batch_size = batch_size

    def store(self, s, a, a_logp, r, next_s, dw, done):

        assert self.count < self.batch_size

        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logp[self.count] = a_logp
        self.r[self.count] = r
        self.next_s[self.count] = next_s
        self.done[self.count] = done
        self.count += 1

    def get_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logp = torch.tensor(self.a_logp, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        next_s = torch.tensor(self.next_s, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logp, r, next_s, done


class PPO:
    def __init__(self, xdim, udim, hdim=32, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lambd=0.97, K_epochs=6, eps_clip=0.2):

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Trick: eps = 1e-5
        self.actor = PolicyNet(xdim, udim, hdim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=1e-5)

        self.critic = ValueNet(xdim, hdim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5)

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

        def normalize_tensor(x):
            mean = x.mean()
            std = x.std()
            normalized_x = (x - mean) / std
            return normalized_x

        # TODO: add buffer
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
        assert s.shape[0] == a.shape[0] == r.shape[0] == next_s.shape[0] == done.shape[0] == 1000

        TD = r + self.gamma * self.critic(next_s) * (1 - done)
        delta = TD - self.critic(s)

        reward_to_go = compute_advantage(
            r, self.gamma)  # Compute reward to go on CPU

        # GAE advantage
        advantage = compute_advantage(
            delta, self.gamma * self.lambd)  # Compute advantage on CPU

        assert advantage.shape == done.shape

        # Trick: advantage normalization
        advantage = normalize_tensor(advantage)

        mu, std = self.actor(s)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(a)

        for _ in range(self.K_epochs):
            mu, std = self.actor(s)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(a)
            ratio = torch.exp(log_probs - old_log_probs)

            option1 = ratio * advantage
            option2 = torch.clamp(ratio, 1 - self.eps_clip,
                                  1 + self.eps_clip) * advantage
            actor_loss = -torch.min(option1, option2).mean()

            # penalty = self.lambd * (ratio - 1.0) ** 2
            # actor_loss = -(ratio * advantage - penalty).mean()

            critic_loss = F.mse_loss(self.critic(s), reward_to_go.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            # Trick: gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

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

        # TODO: input normalization
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

        if (i+1) % 10 == 0:
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

    def moving_average(data, window_size):
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_size:] -
                  cumsum_vec[:-window_size]) / window_size
        return ma_vec

    actor_path = f'./{task_name}_model/ppo_actor_1000.pth'
    critic_path = f'./{task_name}_model/ppo_critic_1000.pth'
    return_list_path = f'./{task_name}_model/return_list_1000.pkl'

    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))

    with open(return_list_path, 'rb') as f:
        loaded_return_list = pickle.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(loaded_return_list, label='Episode Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'{task_name.capitalize()} Task - Return per Episode Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{task_name}_return_plot.png')
    plt.show()

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

    agent = train(env, agent, 1000)
    task_name = 'walk_submit'  # 或 'stand'
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
