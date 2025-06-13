from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]

from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_object, load_object, create_radar_score_baseline, calculate_radar_score, plot_radar_chart, plot_metric_against_baseline

import os
import random
import logging
import pandas as pd
import numpy as np
import torch
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from trademaster.environments.portfolio_management.sarl_environment import PortfolioManagementSARLEnvironment

logging.disable(logging.INFO)
logging.disable(logging.WARNING)
ray.init(ignore_reinit_error=True)

def env_creator(env_config):
    return PortfolioManagementSARLEnvironment(env_config)
register_env("portfolio_management_sarl", env_creator)

from ray.tune.registry import register_env
from trademaster.environments.portfolio_management import PortfolioManagementSARLEnvironment  # or your path

register_env("portfolio_management_sarl", lambda config: PortfolioManagementSARLEnvironment(config))



@TRAINERS.register_module()
class PortfolioManagementSARLTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__()

        self.device = get_attr(kwargs, "device", None)
        self.configs = get_attr(kwargs, "configs", {})
        self.agent_name = get_attr(kwargs, "agent_name", "SAC").upper()
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.dataset = get_attr(kwargs, "dataset", None)
        self.work_dir = os.path.join(ROOT, get_attr(kwargs, "work_dir", ""))
        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)
        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.num_threads = int(get_attr(kwargs, "num_threads", 8))
        self.verbose = get_attr(kwargs, "verbose", False)

        self.trainer_cls = self.select_algorithm(self.agent_name)
        self.setup_rllib_config()
        self.init_before_training()

    def select_algorithm(self, alg_name):
        if alg_name == "SAC":
            from ray.rllib.algorithms.sac import SAC as trainer
        else:
            raise NotImplementedError(f"Algorithm {alg_name} not supported.")
        return trainer

    def setup_rllib_config(self):
        if self.dataset is None:
            raise ValueError("Dataset is required but not provided")

        env_config = {
            "dataset": self.dataset,
            "task": "train"
        }

        self.config = (
        SACConfig()
        .environment(env="portfolio_management_sarl", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=1)
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .training(
            gamma=0.99,
            tau=0.005,
            train_batch_size=256,
            n_step=1,
            target_entropy="auto",
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4
        )
        .debugging(seed=self.random_seed)
    )

    # ✅ Corrected: use self.config instead of undefined `config`
        self.config.model["Q_model"] = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        }
        self.config.model["policy_model"] = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        }

        if self.configs:
            self.config.update_from_dict(self.configs)

    # ✅ Build the trainer instance from config
        self.trainer = self.config.build()


    def init_before_training(self):
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        if self.if_remove is None:
            self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.work_dir}? ") == 'y')

        if self.if_remove:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            if self.verbose:
                logging.info(f"| Removed work_dir: {self.work_dir}")
        else:
            if self.verbose:
                logging.info(f"| Keep work_dir: {self.work_dir}")

        os.makedirs(self.work_dir, exist_ok=True)
        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        os.makedirs(self.checkpoints_path, exist_ok=True)

    def train_and_valid(self):
        valid_score_list = []
        save_dict_list = []

        self.trainer = self.trainer_cls(config=self.configs.to_dict())

        for epoch in range(1, self.epochs + 1):
            logging.info(f"Train Episode: [{epoch}/{self.epochs}]")
            self.trainer.train()

            valid_config = {"dataset": self.dataset, "task": "valid"}
            self.valid_environment = env_creator(valid_config)

            logging.info(f"Valid Episode: [{epoch}/{self.epochs}]")
            state = self.valid_environment.reset()
            episode_reward_sum = 0
            done = False
            while not done:
                action = self.trainer.inference().compute_single_action(state)
                state, reward, done, information = self.valid_environment.step(action)
                episode_reward_sum += reward

            save_dict_list.append(information)
            valid_score_list.append(information.get("sharpe_ratio", 0))
            logging.info(information.get('table', 'No info table'))

            checkpoint_path = os.path.join(self.checkpoints_path, f"checkpoint-{epoch:05d}.pkl")
            checkpoint_obj = self.trainer.save_to_object()
            save_object(checkpoint_obj, checkpoint_path)

        max_index = np.argmax(valid_score_list)
        best_info = save_dict_list[max_index]

        plot_metric_against_baseline(
            total_asset=best_info['total_assets'],
            buy_and_hold=None,
            alg='SARL',
            task='valid',
            color='darkcyan',
            save_dir=self.work_dir
        )

        best_checkpoint = os.path.join(self.checkpoints_path, f"checkpoint-{max_index+1:05d}.pkl")
        best_obj = load_object(best_checkpoint)
        save_object(best_obj, os.path.join(self.checkpoints_path, "best.pkl"))

    def test(self):
        self.trainer = self.trainer_cls(config=self.configs.to_dict())
        obj = load_object(os.path.join(self.checkpoints_path, "best.pkl"))
        self.trainer.restore_from_object(obj)

        test_config = {"dataset": self.dataset, "task": "test"}
        self.test_environment = env_creator(test_config)

        logging.info("Test Best Episode")
        state = self.test_environment.reset()
        episode_reward_sum = 0
        done = False
        while not done:
            action = self.trainer.inference().compute_single_action(state)
            state, reward, done, sharpe = self.test_environment.step(action)
            episode_reward_sum += reward
            if done:
                plot_metric_against_baseline(
                    total_asset=sharpe['total_assets'],
                    buy_and_hold=None,
                    alg='SARL',
                    task='test',
                    color='darkcyan',
                    save_dir=self.work_dir
                )
                break

        logging.info(sharpe.get('table', 'No table info'))

        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values

        df = pd.DataFrame({
            "daily_return": daily_return,
            "total assets": assets
        })
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)

    def dynamics_test(self, test_dynamic, cfg):
        self.trainer = self.trainer_cls(config=self.configs.to_dict())
        obj = load_object(os.path.join(self.checkpoints_path, "best.pkl"))
        self.trainer.restore_from_object(obj)

        test_dynamic_envs = []
        for i, path in enumerate(self.dataset.test_dynamic_paths):
            config = {
                "dataset": self.dataset,
                "task": "test_dynamic",
                "test_dynamic": test_dynamic,
                "dynamics_test_path": path,
                "task_index": i,
                "work_dir": cfg.work_dir,
            }
            test_dynamic_envs.append(env_creator(config))

        def Average_holding(states, env, weights_brandnew):
            if weights_brandnew is None:
                return [0] + [1 / env.stock_dim] * env.stock_dim
            return weights_brandnew

        def Do_Nothing(states, env):
            return [1] + [0] * env.stock_dim

        def test_single_env(env, policy, policy_id=None):
            env.test_id = policy_id
            state = env.reset()
            done = False
            weights_brandnew = None
            while not done:
                if policy_id == "Average_holding":
                    action = policy(state, env, weights_brandnew)
                elif policy_id == "Do_Nothing":
                    action = policy(state, env)
                else:
                    action = policy(state)
                state, reward, done, info = env.step(action)
                weights_brandnew = info.get("weights_brandnew", None)
            rewards = env.save_asset_memory()
            assets = rewards["total assets"].values
            df_return = env.save_portfolio_return_memory()
            daily_return = df_return.daily_return.values
            df = pd.DataFrame({
                "daily_return": daily_return,
                "total assets": assets
            })
            df.to_csv(os.path.join(self.work_dir, f"test_dynamic_result_style_{test_dynamic}_part_{env.test_id}.csv"), index=False)
            return daily_return

        for i, env in enumerate(test_dynamic_envs):
            test_single_env(env, self.trainer.compute_single_action, 'agent')
            test_single_env(env, Average_holding, 'Average_holding')
            test_single_env(env, Do_Nothing, 'Do_Nothing')

        metric_path = f'metric_test_dynamic_{cfg.data.test_dynamic}'
        metrics_sigma_dict, zero_metrics = create_radar_score_baseline(cfg.work_dir, metric_path, 'Do_Nothing', 'Average_holding')
        test_metrics_scores_dict = calculate_radar_score(cfg.work_dir, metric_path, 'agent', metrics_sigma_dict, zero_metrics)
        plot_radar_chart(test_metrics_scores_dict, f'radar_plot_agent_{test_dynamic}.png', cfg.work_dir)

        logging.info("Dynamics test completed.")
