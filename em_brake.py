import gym
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

class EmergencyBraking(gym.Env):
    def __init__(self):
        metadata = {'render.modes': ['human']}
        self.step_length = 0.1  # ms
        self.action_number = 1
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.full([2,], -float('inf')),np.full([2,], float('inf')), dtype=np.float32)
        self.obs = self._reset_init_state()
        self.A = np.array([[1, -self.step_length],[0,1]])
        self.B = np.array([[0],[self.step_length]])


    def reset(self):
        self.obs = self._reset_init_state()
        return self.obs

    def step(self, action):
        if len(action.shape) == 2:
            action = action.reshape([-1,])
        self.action = self._action_transform(action)
        reward = self.compute_reward(self.obs, self.action)
        self.obs = (np.matmul(self.A, self.obs[:, np.newaxis]) + np.matmul(self.B, self.action[:, np.newaxis])).reshape([-1,])
        if self.obs[1] < 0: self.obs[1] = 0
        done = True if self.obs[0] < 0 or self.obs[1] <= 0 else False
        info = dict()
        return self.obs, reward, done, info



    def _action_transform(self, action):
        action = np.clip(action, 5, -5)
        return action

    def compute_reward(self, obs, action):
        r = 0.01*(obs[0] ** 2 + obs[1] ** 2 + action ** 2)
        return r

    def _reset_init_state(self):
        d = 10 * np.random.random()
        v = 10 * np.random.random()
        return np.array([d,v])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        extension = 1
        if mode == 'human':
            plt.ion()
            plt.cla()
            plt.title("Emergency Braking")
            ax = plt.axes(xlim=(-7 - extension, 5 + extension),
                          ylim=(-1, 5))
            plt.axis("equal")
            plt.axis('off')

            plt.plot([-7, 5], [0, 0])
            plt.plot([5, 5], [0, 3])
            ax.add_patch(plt.Rectangle((-self.obs[0]+5-2, 0), 2, 1, edgecolor='black', facecolor='none'))

            text_x, text_y = -5, 5
            plt.text(text_x, text_y, 'distance: {:.2f}m'.format(self.obs[0]))
            plt.text(text_x, text_y - 1, 'velocity: {:.2f}m'.format(self.obs[1]))
            plt.text(text_x, text_y - 2, 'action: {:.2f}m'.format(self.action[0]))
            plt.show()
            plt.pause(0.001)

def test_env():
    env = EmergencyBraking()
    obs = env.reset()
    i = 0
    done = 0
    action = np.array([-5])
    while True:
        obs, reward, done, info = env.step(action)
        env.render()
        if done: env.reset()
        print(reward)

if __name__ == '__main__':
    test_env()










