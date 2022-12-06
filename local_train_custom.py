import numpy as np
from datetime import datetime

import wandb

from agents.orderenforcingwrapper_train import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv


run_label = "train"
project_name = "SAC"

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb.config = {
    "learning_rate": "3e-4",
    "batch_size": 512,
    "episodes": 250,
    "reward": "",
    "label": run_label,
    "start_datetime": start_datetime,
}

run_name = f"{run_label}_{start_datetime}"
wandb.init(project=project_name, name=run_name, entity="m509ua", config=wandb.config)


class Constants:
    episodes = wandb.config["episodes"]
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'


def action_space_to_dict(
        aspace
):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations}
    return obs_dict



def train():
    print("Starting local train")
    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    obs_dict = env_reset(env)

    actions = agent.register_reset(obs_dict)

    episodes_completed = 0
    num_steps = 0
    episode_metrics = []
    save_metrics = []


    while True:
        observations, rewards, done, _ = env.step(actions)

        actions = agent.compute_action(observations, done, rewards)

        if done:
            episodes_completed += 1

            metrics_t = env.evaluate()
            metrics = {
                "p_cost": round(metrics_t[0], 5),
                "e_cost": round(metrics_t[1], 5),
                "g_cost": round(metrics_t[2], 5),
                "avg": round((metrics_t[0] + metrics_t[1] + metrics_t[2]) / 3, 5)
            }
            if np.any(np.isnan(metrics_t)):
                raise ValueError("Episode metrics are nan, please content organizers")
            episode_metrics.append(metrics)
            save_metrics.append(metrics)
            print(f"Episode: {episodes_completed} | Num Steps: {num_steps} | metrics: {metrics}", )

            env_label = ""
            wandb.log({f"scores{env_label}/p_cost": round(metrics_t[0], 5)}, step=episodes_completed)
            wandb.log({f"scores{env_label}/e_cost": round(metrics_t[1], 5)}, step=episodes_completed)
            #wandb.log({f"scores{env_label}/g_cost": round(metrics_t[2], 5)}, step=episodes_completed)
            wandb.log({f"scores{env_label}/pe_avg": round((metrics_t[0] + metrics_t[1]) / 2, 5)}, step=episodes_completed)
            wandb.log({"episode": episodes_completed}, step=episodes_completed)
            wandb.log({"num_steps": num_steps}, step=episodes_completed)

            obs_dict = env_reset(env)
            actions = agent.register_reset(obs_dict)

        num_steps += 1

        if episodes_completed >= Constants.episodes:
            break

if __name__ == '__main__':
    train()
