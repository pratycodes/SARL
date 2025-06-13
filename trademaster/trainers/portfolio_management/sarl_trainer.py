from pathlib import Path
import os
import random
import logging

import torch
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from trademaster.environments.portfolio_management.sarl_environment import PortfolioManagementSARLEnvironment
from trademaster.utils import get_attr, save_object, load_object, create_radar_score_baseline, calculate_radar_score, plot_radar_chart, plot_metric_against_baseline
from ..custom import Trainer
from ..builder import TRAINERS

# Register your environment
def env_creator(env_config):
    return PortfolioManagementSARLEnvironment(env_config)

register_env("PortfolioManagementSARL-v0", env_creator)


@TRAINERS.register_module()
class PortfolioManagementSARLTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = get_attr(kwargs, "device", None)
        self.configs = get_attr(kwargs, "configs", {})
        self.agent_name = get_attr(kwargs, "agent_name", "ppo").lower()
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.dataset = get_attr(kwargs, "dataset", None)
        self.work_dir = os.path.join(Path(__file__).resolve().parents[3], get_attr(kwargs, "work_dir", ""))
        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)
        self.num_threads = int(get_attr(kwargs, "num_threads", 8))
        self.verbose = get_attr(kwargs, "verbose", False)

        self.init_before_training()

    def init_before_training(self):
        # seeding
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_default_dtype(torch.float32)

        # clear logs
        if self.if_remove:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)
        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        os.makedirs(self.checkpoints_path, exist_ok=True)
        ray.init(ignore_reinit_error=True)

    def _build_trainer(self, task: str):
        cfg = PPOConfig()
        cfg = (
            cfg.environment(
                env="PortfolioManagementSARL-v0",
                env_config={"dataset": self.dataset, "task": task}
            )
            .framework("torch")
            .resources(num_gpus=self.configs.get("num_gpus", 0))
            .env_runners(num_env_runners=self.configs.get("num_workers", 0))  # <- ✅ FIX
            .training(
                lr=self.configs.get("lr", 5e-5),
                train_batch_size=self.configs.get("train_batch_size", 4000),
                gamma=self.configs.get("gamma", 0.99),
                model=self.configs.get("model", {}),
            )
        )
        return cfg.build()


    def train_and_valid(self):
        valid_scores, save_dicts = [], []
        self.trainer = self._build_trainer(task="train")

        for epoch in range(1, self.epochs + 1):
            print(f"Training Epoch {epoch}/{self.epochs}")
            self.trainer.train()

            print(f"Validating Epoch {epoch}/{self.epochs}")
            valid_trainer = self._build_trainer(task="valid")
            episode_reward = 0
            obs, _ = valid_trainer.workers.local_worker().env.reset()
            done = False
            while not done:
                action = valid_trainer.compute_single_action(obs, explore=False)
                obs, reward, done, truncated, info = valid_trainer.workers.local_worker().env.step(action)
                episode_reward += reward

            save_dicts.append(info)
            valid_scores.append(info["sharpe_ratio"])

            path = self.trainer.save()
            os.makedirs(self.checkpoints_path, exist_ok=True)
            os.replace(path, os.path.join(self.checkpoints_path, f"checkpoint_{epoch:05d}"))

        best_idx = int(np.argmax(valid_scores)) + 1
        plot_metric_against_baseline(
            total_asset=save_dicts[best_idx - 1]["total_assets"], buy_and_hold=None,
            alg='SARL', task='valid', color='darkcyan', save_dir=self.work_dir
        )

        # copy best checkpoint to 'best'
        best_cp = os.path.join(self.checkpoints_path, f"checkpoint_{best_idx:05d}")
        os.replace(best_cp, os.path.join(self.checkpoints_path, "best"))

        ray.shutdown()

    def test(self):
        self.trainer = PPOConfig().environment(env="PortfolioManagementSARL-v0", env_config={"dataset": self.dataset, "task": "test"}).framework("torch").build()
        self.trainer.restore(os.path.join(self.checkpoints_path, "best"))

        print("Testing on best policy")
        obs, _ = self.trainer.workers.local_worker().env.reset()
        done = False
        while not done:
            action = self.trainer.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = self.trainer.workers.local_worker().env.step(action)

        plot_metric_against_baseline(
            total_asset=info["total_assets"], buy_and_hold=None,
            alg='SARL', task='test', color='darkcyan', save_dir=self.work_dir
        )
        self.trainer.workers.local_worker().env.save_asset_memory().to_csv(os.path.join(self.work_dir, "test_result.csv"))
        ray.shutdown()

    # dynamics_test remains unchanged, using self.trainer similarly
