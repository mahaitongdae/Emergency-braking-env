import numpy as np
import tensorflow as tf

class EmBrakeModel(object):
    def rollout_out(self):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)
            # self.reward_info.update({'final_rew': rewards.numpy()[0]})

            return self.obses, rewards, constraints

    def compute_reward(self, obs, action):
        r = 0.01*(tf.square(obs[:,0]) + tf.square(obs[:,1]) + tf.square(action[:,0]))
        constraints = tf.where(obs[:,0]>0, obs[:,0], tf.zeros_like(obs[:,0]))
        return r, constraints
    
    def _action_transformation_for_end2end(actions):
        actions = tf.clip_by_value(actions, -1.05, 1.05)
        acc = 5 * actions
        return actions
    
    def f_xu(self, x, u, frequency=10.0):
        d, v = x[:, 0], x[:, 1]
        a = u[:, 0]
        next_state = [d- 1/frequency*v, v+1/frequency*a]
        return tf.stack(next_state, 1)
    
    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None
        
        