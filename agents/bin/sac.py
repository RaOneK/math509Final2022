import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List
from agents.bin.rlc import RLC
from agents.bin.buffer import PrioritizedReplay
from agents.bin.rl import Actor, Critic, SoftQNetwork


class SAC(RLC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # internally defined
        self.__normalized = False
        self.__alpha = 0.2
        self.actor_update_freq = 1
        self.reward_scaling = 10.0
        # self.decoder_latent_lambda = 1e-6
        self.__soft_q_criterion = nn.SmoothL1Loss()
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__replay_buffer = PrioritizedReplay(int(self.replay_buffer_capacity))
        self.__soft_q_net1 = None
        self.__soft_q_net2 = None
        self.__target_soft_q_net1 = None
        self.__target_soft_q_net2 = None
        self.__policy_net = None
        self.__soft_q_optimizer1 = None
        self.__soft_q_optimizer2 = None
        self.__policy_optimizer = None
        self.__target_entropy = None
        self.__norm_mean = None
        self.__norm_std = None
        self.__r_norm_mean = None
        self.__r_norm_std = None

        self.__set_networks()

    @property
    def device(self) -> torch.device:
        """Device; cuda or cpu."""

        return self.__device

    @property
    def soft_q_net1(self) -> SoftQNetwork:
        """soft_q_net1."""

        return self.__soft_q_net1

    @property
    def soft_q_net2(self) -> SoftQNetwork:
        """soft_q_net2."""

        return self.__soft_q_net2

    @property
    def policy_net(self) -> Actor:
        """policy_net."""

        return self.__policy_net

    @property
    def norm_mean(self) -> List[float]:
        """norm_mean."""

        return self.__norm_mean

    @property
    def norm_std(self) -> List[float]:
        """norm_std."""

        return self.__norm_std

    @property
    def normalized(self) -> bool:
        """normalized."""

        return self.__normalized

    @property
    def r_norm_mean(self) -> float:
        """r_norm_mean."""

        return self.__r_norm_mean

    @property
    def r_norm_std(self) -> float:
        """r_norm_std."""

        return self.__r_norm_std

    @property
    def replay_buffer(self) -> PrioritizedReplay:
        """replay_buffer."""

        return self.__replay_buffer

    @property
    def alpha(self) -> float:
        """alpha."""

        return self.__alpha
        # return self.log_alpha.exp()

    @property
    def soft_q_criterion(self) -> nn.SmoothL1Loss:
        """soft_q_criterion."""

        return self.__soft_q_criterion

    @property
    def soft_q_criterion1(self) -> nn.MSELoss:
        """soft_q_criterion."""

        return self.__soft_q_criterion1

    @property
    def target_soft_q_net1(self) -> SoftQNetwork:
        """target_soft_q_net1."""

        return self.__target_soft_q_net1

    @property
    def target_soft_q_net2(self) -> SoftQNetwork:
        """target_soft_q_net2."""

        return self.__target_soft_q_net2

    @property
    def soft_q_optimizer1(self) -> optim.Adam:
        """soft_q_optimizer1."""

        return self.__soft_q_optimizer1

    @property
    def soft_q_optimizer2(self) -> optim.Adam:
        """soft_q_optimizer2."""

        return self.__soft_q_optimizer2

    @property
    def policy_optimizer(self) -> optim.Adam:
        """policy_optimizer."""

        return self.__policy_optimizer

    @property
    def target_entropy(self) -> float:
        """target_entropy."""

        return self.__target_entropy

    def add_to_buffer(
            self,
            observations: List[float],
            actions: List[float],
            reward: float,
            next_observations: List[float],
            done: bool = False,
            agent_id: int = 0
    ):
        """

        :param observations:
        :param actions:
        :param reward:
        :param next_observations:
        :param done:
        :return:
        """
        if self.normalized:
            observations = np.array(self.__get_normalized_observations(observations), dtype=float)
            next_observations = np.array(self.__get_normalized_observations(next_observations), dtype=float)
            reward = self.__get_normalized_reward(reward)
        else:
            pass

        self.__replay_buffer.push(observations, actions, reward, next_observations, done)

        if agent_id == 4:
            if self.time_step >= self.start_training_time_step and self.batch_size <= len(self.__replay_buffer):
                if not self.normalized:
                    X = np.array([j[0] for j in self.__replay_buffer.buffer], dtype=float)
                    self.__norm_mean = np.nanmean(X, axis=0)
                    self.__norm_std = np.nanstd(X, axis=0) + 1e-5
                    # self.__norm_mean = np.array([0.0])
                    # self.__norm_std = np.array([1.0])
                    R = np.array([j[2] for j in self.__replay_buffer.buffer], dtype=float)
                    self.__r_norm_mean = np.nanmean(R, dtype=float)
                    self.__r_norm_std = np.nanstd(R, dtype=float) / self.reward_scaling + 1e-5
                    # self.__r_norm_mean = np.array([0.0])
                    # self.__r_norm_std = np.array([1.0])

                    self.__replay_buffer.buffer = [(
                        np.hstack(
                            (np.array(self.__get_normalized_observations(observations), dtype=float)).reshape(1, -1)[
                                0]),
                        actions,
                        self.__get_normalized_reward(reward),
                        np.hstack(
                            (np.array(self.__get_normalized_observations(next_observations), dtype=float)).reshape(1,
                                                                                                                   -1)[
                                0]),
                        done
                    ) for observations, actions, reward, next_observations, done in self.__replay_buffer.buffer]
                    self.__normalized = True
                else:
                    pass

                k = 1 + len(self.replay_buffer) / self.replay_buffer_capacity
                batch_size = int(k * self.batch_size)
                update_times = int(k * self.update_per_time_step)

                for _ in range(update_times):
                    observations, actions, reward, next_observations, done, idx, weights, next_next_observations = \
                        self.__replay_buffer.sample(batch_size)

                    tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                    observations = tensor(np.array(observations)).to(self.device)
                    next_observations = tensor(np.array(next_observations)).to(self.device)
                    next_next_observations = tensor(np.array(next_next_observations)).to(self.device)
                    actions = tensor(np.array(actions)).to(self.device)
                    reward = tensor(np.array(reward)).unsqueeze(1).to(self.device)
                    done = tensor(np.array(done)).unsqueeze(1).to(self.device)

                    # Update Critic
                    self.update_critic_and_memory(observations, actions, reward, next_observations, 1 - done, idx)

                    # Update Actor
                    if self.time_step % self.actor_update_freq == 0:
                        self.update_actor_and_alpha(observations)

                    soft_update_params(
                        self.critic.Q1, self.critic_target.Q1, self.tau
                    )
                    soft_update_params(
                        self.critic.Q2, self.critic_target.Q2, self.tau
                    )

            else:
                pass

    def update_critic_and_memory(
            self,
            observations,
            actions,
            reward,
            next_observations,
            not_done,
            idx
    ):
        with torch.no_grad():
            new_next_actions, new_log_pi, _ = self.actor.sample(next_observations)
            target_Q1, target_Q2 = self.critic_target(next_observations, new_next_actions)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * new_log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()

        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Update memory
        td_error1 = target_Q.detach() - current_Q1
        td_error2 = target_Q.detach() - current_Q2
        prios = abs(((td_error1 + td_error2) / 2.0 + 1e-5).squeeze())
        self.replay_buffer.update_priorities(idx, prios.data.cpu().numpy())


    def update_actor_and_alpha(
            self,
            observations
    ):
        new_actions, log_pi, _ = self.actor.sample(observations)
        actor_Q1, actor_Q2 = self.critic(observations, new_actions)
        q_new_actions = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.scheduler.step()


    def select_actions(self, observations: List[float], agent_id: int = 0):
        if self.time_step <= self.end_exploration_time_step:
            actions = self.get_exploration_actions()
        else:
            actions = self.__get_post_exploration_actions(observations)

        self.actions = actions

        if agent_id == 4:
            self.next_time_step()

        return actions


    def __get_post_exploration_actions(self, observations: List[float]) -> List[float]:
        """Action sampling using policy, post-exploration time step"""
        with torch.no_grad():
            observations = np.array(self.__get_normalized_observations(observations), dtype=float)
            observations = torch.FloatTensor(observations).unsqueeze(0).to(self.__device)
            actions = self.actor.sample(observations)
            actions = actions[2] if self.time_step >= self.deterministic_start_time_step else actions[0]
            actions = actions.detach().cpu().numpy()[0]
            return actions


    def get_exploration_actions(self) -> List[float]:
        return list(self.action_scaling_coefficient * self.action_space.sample())


    def __get_normalized_reward(self, reward: float) -> float:
        return (reward - self.r_norm_mean) / self.r_norm_std


    def __get_normalized_observations(self, observations: List[float]) -> List[float]:
        return ((np.array(observations, dtype=float) - self.norm_mean) / self.norm_std).tolist()


    def __set_networks(self):
        # init networks
        self.actor = Actor(
            self.observation_dimension,
            self.action_dimension,
            self.action_space,
            self.action_scaling_coefficient,
            self.hidden_dimension,
            # encode_dim=self.encode_dim
        ).to(self.device)
        self.critic = Critic(
            self.observation_dimension,
            self.action_dimension,
            self.hidden_dimension,
            # encode_dim=self.encode_dim
        ).to(self.device)
        self.critic_target = Critic(
            self.observation_dimension,
            self.action_dimension,
            self.hidden_dimension,
            # encode_dim=self.encode_dim
        ).to(self.device)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            99 * 8759 * 2,
            eta_min=self.lr * 0.1
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr
        )
        self.scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            99 * 8759 * 2,
            eta_min=self.lr * 0.1
        )

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.__target_entropy = -np.prod(self.action_space.shape).item()


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )
