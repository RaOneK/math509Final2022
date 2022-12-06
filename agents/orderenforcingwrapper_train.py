import sys
from gym.spaces import Box
from agents.sac_agent import SAC_Agent
from agents.observation_space import observation_space_pre_dicts


def dict_to_action_space(aspace_dict):
    return Box(
        low=aspace_dict["low"],
        high=aspace_dict["high"],
        dtype=aspace_dict["dtype"],
    )


class OrderEnforcingAgent:
    """
    TRY NOT TO CHANGE THIS

    Emulates order enforcing wrapper in Pettingzoo for easy integration
    Calls each agent step with agent in a loop and returns the action
    """

    def __init__(self):
        self.time_step = 0
        self.time_steps = 8760
        self.agent = SAC_Agent()
        self.action_space = None
        self.num_buildings = None
        self.previous_actions = None
        self.observation_space = None
        self.previous_observations = None

    def register_reset(self, observation):
        """Get the first observation after env.reset, return action"""
        action_space = observation["action_space"]
        self.action_space = [dict_to_action_space(asd) for asd in action_space]
        self.observation_space = [observation_space_pre_dicts for _ in range(len(action_space))]
        obs = observation["observation"]
        self.num_buildings = len(obs)

        actions = []

        for agent_id in range(self.num_buildings):
            actions.append(
                self.agent.register_reset(
                    observation=obs[agent_id],
                    action_space=self.action_space[agent_id],
                    observation_space=self.observation_space[agent_id],
                    agent_id=agent_id,
                )
            )

        self.previous_actions = actions
        self.previous_observations = obs
        self.time_step = 1
        return actions

    def compute_action(self, observation, done=False, rewards=None):
        """Get observation return action"""
        assert self.num_buildings is not None

        actions = []

        if self.judge_done(observation[0]):
            done = True

        for agent_id in range(self.num_buildings):
            reward = rewards[agent_id]
            self.agent.update_policy(
                previous_observation=self.previous_observations[agent_id],
                action=self.previous_actions[agent_id],
                reward=reward,
                observation=observation[agent_id],
                done=done,
                agent_id=agent_id
            )
            actions.append(self.agent.compute_action(observation[agent_id], agent_id))

        self.previous_actions = actions
        self.previous_observations = observation
        self.time_step += 1
        return actions

    @staticmethod
    def judge_done(
            observation
    ):
        if int(observation[1]) == 7 and int(observation[2]) == 24:
            return True
        else:
            return False

