import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ''' 
            TODO #24 ✅ : return the action that maxinmizes the Q-value .
        '''
        return self.critic.qa_values(observation).argmax(-1).squeeze()