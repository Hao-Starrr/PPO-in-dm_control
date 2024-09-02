import util
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
    def __init__(self, xdim, udim, batch_size, gamma=0.99, lambd=0.97):
        # direct from env
        self.s = np.zeros((batch_size, xdim), dtype=np.float32)
        self.r = np.zeros(batch_size, dtype=np.float32)

        # calculated from actor
        self.a = np.zeros((batch_size, udim), dtype=np.float32)
        self.a_logp = np.zeros((batch_size, udim), dtype=np.float32)

        # calculated from critic
        self.value = np.zeros(batch_size, dtype=np.float32)

        # calculated in buffer
        self.advantage = np.zeros(batch_size, dtype=np.float32)
        self.reward_to_go = np.zeros(batch_size, dtype=np.float32)
        self.gamma, self.lambd = gamma, lambd
        self.count, self.path_start_idx, self.batch_size = 0, 0, batch_size

    def store(self, s, a, r, value, logp):

        assert self.count < self.batch_size

        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.value[self.count] = value
        self.a_logp[self.count] = logp
        self.count += 1

    def finish_path(self, last_value=0):

        path_slice = slice(self.path_start_idx, self.count)
        r = np.append(self.r[path_slice], last_value)
        value = np.append(self.value[path_slice], last_value)

        # the next two lines implement GAE-Lambda advantage calculation
        delta = r[:-1] + self.gamma * value[1:] - value[:-1]
        self.advantage[path_slice] = util.discount_cumsum(
            delta, self.gamma * self.lambd)

        # the next line computes rewards-to-go, to be targets for the value function
        self.reward_to_go[path_slice] = util.discount_cumsum(r, self.gamma)[
            :-1]

        self.path_start_idx = self.count

    def get_tensor(self):

        # buffer has to be full before you can get
        assert self.count == self.batch_size

        self.count, self.path_start_idx = 0, 0

        # advantage normalization
        adv_mean, adv_std = self.advantage.mean(), self.advantage.std()
        self.advantage = (self.advantage - adv_mean) / adv_std

        data = dict(s=self.s, a=self.a, reward_to_go=self.reward_to_go,
                    advantage=self.advantage, a_logp=self.a_logp)
        return {key: torch.as_tensor(val, dtype=torch.float32) for key, val in data.items()}


class PPO:
    def __init__(self, xdim, udim, hdim=32, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lambd=0.97, K_epochs=10, eps_clip=0.2):

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
        value = self.critic(state)

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0], logp.detach().cpu().numpy()[0]

    def update(self, buffer):

        s = buffer['s'].to(self.device)
        a = buffer['a'].to(self.device)
        reward_to_go = buffer['reward_to_go'].view(-1, 1).to(self.device)
        advantage = buffer['advantage'].view(-1, 1).to(self.device)
        logp = buffer['a_logp'].to(self.device)

        # assert s.shape == (1000, xdim)
        # assert a.shape == (1000, udim)
        # assert reward_to_go.shape == (1000, 1)
        # assert advantage.shape == (1000, 1)
        # assert logp.shape == (1000, udim)

        # Trick: advantage normalization
        advantage = util.normalize(advantage)

        old_log_probs = logp.detach()

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
            critic_loss = F.mse_loss(self.critic(s), reward_to_go.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            # Trick: gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

            self.actor_optimizer.step()
            self.critic_optimizer.step()


def train(env, agent, buffer, num_epochs, buffer_size):
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    T = buffer_size

    return_list = []

    episode_return = 0
    t = env.reset()
    # state
    x = t.observation
    x = np.array(x['orientations'].tolist() +
                 [x['height']]+x['velocity'].tolist())
    # Trick: state normalization
    x = util.normalize(x)
    # done
    done = False

    for i in range(num_epochs):
        for timestep in range(T):

            u, value, logp = agent.select_action(x)
            r = env.step(u)

            # state
            next_x = r.observation
            next_x = np.array(next_x['orientations'].tolist() +
                              [next_x['height']]+next_x['velocity'].tolist())
            next_x = util.normalize(next_x)
            # reward
            # TODO: reward scaling
            reward = r.reward
            # done
            done = r.last()

            assert x.shape == (xdim,)
            assert u.shape == (udim,)
            assert next_x.shape == (xdim,)
            assert reward.shape == ()
            assert done == True or done == False

            buffer.store(x, u, reward, value, logp)

            episode_return += reward

            x = next_x

            if done or timestep == T-1:

                # case 1: done in the game
                if done:
                    buffer.finish_path(last_value=0)
                    return_list.append(episode_return)
                    print(f'Episode {i}: return {episode_return}')
                # case 2: the buffer is full
                elif timestep == T-1:
                    buffer.finish_path(last_value=agent.critic(
                        torch.FloatTensor(next_x).unsqueeze(0).to(device)))
                    print("buffer is full, finishing path")

                # initialize the next episode
                episode_return = 0
                t = env.reset()
                # state
                x = t.observation
                x = np.array(x['orientations'].tolist() +
                             [x['height']]+x['velocity'].tolist())
                # Trick: state normalization
                x = util.normalize(x)

        agent.update(buffer.get_tensor())
        print("~~~~update~~~~")

    return agent


def load_model(agent, task_name, window_size=100):

    def moving_average(data, window_size):
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_size:] -
                  cumsum_vec[:-window_size]) / window_size
        return ma_vec

    actor_path = f'./{task_name}_model/ppo_actor_5000.pth'
    critic_path = f'./{task_name}_model/ppo_critic_5000.pth'
    return_list_path = f'./{task_name}_model/return_list_5000.pkl'

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

    buffer_size = 1000 * 2
    agent = PPO(xdim, udim)
    buffer = ReplayBuffer(xdim, udim, batch_size=buffer_size)

    agent = train(env, agent, buffer, num_epochs=500, buffer_size=buffer_size)
    task_name = 'walk'  # 或 'stand'
    agent = load_model(agent, task_name)

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
