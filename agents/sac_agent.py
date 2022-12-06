import numpy as np

from gym.spaces import Box
from agents.bin.sac import SAC
from agents.tool.normalization import periodic_normalization, normalize


def dict_to_action_space(aspace_dict):
    return Box(
        low=aspace_dict["low"],
        high=aspace_dict["high"],
        dtype=aspace_dict["dtype"],
    )


class SAC_Agent:
    def __init__(self):
        self.action_space = {}
        self.observation_space = {}
        self.agent = None
        self.encoder = None
        self.observation = {}
        self.time_step = {}

    def register_reset(
            self,
            observation,
            action_space,
            observation_space,
            agent_id,
    ):
        """Get the first observation after env.reset, return action"""
        self.action_space[agent_id] = action_space
        observation, observation_space = self.get_states(
            observation,
            observation_space,
        )

        self.observation_space[agent_id] = observation_space
        self.time_step[agent_id] = 0

        if self.agent is None:
            self.agent = SAC(
                action_space=self.action_space[agent_id],
                observation_space=observation_space,
            )

        self.observation[agent_id] = observation
        return self.agent.select_actions(observation)

    def compute_action(
            self,
            observation,
            agent_id
    ):
        """Get observation return action"""
        self.time_step[agent_id] += 1

        return self.agent.select_actions(self.observation[agent_id], agent_id=agent_id)

    def update_policy(
            self,
            previous_observation,
            action,
            reward,
            observation,
            done,
            agent_id
    ):
        """Update buffer and policy"""
        observation, observation_space = self.get_states(observation, self.observation_space)
        self.agent.add_to_buffer(self.observation[agent_id], action, reward, observation, done, agent_id)
        self.observation[agent_id] = observation


    def get_states(
            self,
            observation,
            observation_space
    ):
        """
        Get states
        """
        if not self.encoder:
            self.encoder = self.get_encoder(observation_space)

        if len(self.encoder) == 0:
            pass
        else:
            observation = np.hstack([e * o for o, e in zip(observation, self.encoder) if e is not None])
            observation_space['shape'] = (observation.size,)
        return observation, observation_space


    @staticmethod
    def get_encoder(
            observation_space,
    ):
        """
        Get encoder
        """
        obs_encode = []
        if observation_space['shape'] == (28,):
            index = 0
            assert(len(observation_space['high']) == len(observation_space['low']))

            for above, below in zip(observation_space['high'], observation_space['low']):
                if index in [0, 1, 2]:  # hour, daytype, month
                    obs_encode.append(periodic_normalization(above))
                else:
                    obs_encode.append(normalize(below, above))
                index += 1

        return obs_encode
