from __future__ import annotations
import torch
import sys
from pathlib import Path
import random

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from collections import OrderedDict
import pickle
import os.path as osp

from trademaster.utils import get_attr, print_metrics
from ..custom import Environments
from ..builder import ENVIRONMENTS
from trademaster.pretrained import pretrained
from gymnasium import spaces
from trademaster.nets import mLSTMClf


@ENVIRONMENTS.register_module()
class PortfolioManagementSARLEnvironment(Environments):
    def __init__(self, config):
        super(PortfolioManagementSARLEnvironment, self).__init__()
        self.dataset = get_attr(config, "dataset", None)
        self.task = get_attr(config, "task", "train")
        self.test_dynamic = int(get_attr(config, "test_dynamic", "-1"))
        self.task_index = int(get_attr(config, "task_index", "-1"))
        self.work_dir = get_attr(config, "work_dir", "")
        length_day = get_attr(self.dataset, "length_day", 10)
        self.day = length_day
        self.df_path = None

        if self.task.startswith("train"):
            self.df_path = get_attr(self.dataset, "train_path", None)
        elif self.task.startswith("valid"):
            self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            self.df_path = get_attr(self.dataset, "test_path", None)

        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.001)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])

        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(config, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
            self.start_date = self.df.loc[:, 'date'].iloc[0]
            self.end_date = self.df.loc[:, 'date'].iloc[-1]
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)

        self.stock_dim = len(self.df.tic.unique())
        self.tic_list = self.df.tic.unique()
        self.state_space_shape = self.stock_dim
        self.action_space_shape = self.stock_dim + 1
        self.length_day = length_day

        # Load pre-trained mLSTM classifier
        self.network_dict = torch.load(get_attr(pretrained, "sarl_encoder", None), map_location=torch.device('cpu'))
        self.net = mLSTMClf(
            n_features=len(self.tech_indicator_list),
            layer_num=1,
            n_hidden=128,
            tic_number=len(self.tic_list)
        ).to("cpu")
        self.net.load_state_dict(self.network_dict)

        self.action_space = spaces.Box(
            low=-5,
            high=5,
            shape=(self.action_space_shape,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=((len(self.tech_indicator_list) + 1) * self.state_space_shape,),
            dtype=np.float32
        )

        self._initialize_state()
        self.test_id = 'agent'

    def _initialize_state(self):
        self.data = self.df.loc[self.day, :]
        tic_list = list(self.data.tic)

        s_market = np.array([
            self.data[tech].values.tolist()
            for tech in self.tech_indicator_list
        ]).reshape(-1).astype(np.float32).tolist()

        X = []
        for tic in tic_list:
            df_tic = self.df[self.df.tic == tic]
            df_information = df_tic[self.day - self.length_day:self.day][self.tech_indicator_list].to_numpy()
            df_information = torch.from_numpy(df_information).float().unsqueeze(0)
            X.append(df_information)

        X = torch.cat(X, dim=0).unsqueeze(0).to("cpu")
        y = self.net(X).cpu().detach().squeeze().numpy().astype(np.float32).tolist()

        self.state = np.array(s_market + y, dtype=np.float32)
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1 / self.action_space_shape] * self.action_space_shape]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []
        self.reward = np.float32(0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.day = self.length_day
        self._initialize_state()
        return self.state, {}

    def step(self, actions):
        actions = np.array(actions)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        terminated = self.terminal
        truncated = False  # modify if using episode timeout

        if self.terminal:
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date} to {self.end_date}')
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            stats = OrderedDict({
                "Total Return": ["{:04f}%".format(tr * 100)],
                "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                "Volatility": ["{:04f}%".format(vol * 100)],
                "Max Drawdown": ["{:04f}%".format(mdd * 100)],
            })
            table = print_metrics(stats)

            df_return = self.save_portfolio_return_memory()
            daily_return = df_return.daily_return.values
            df_value = self.save_asset_memory()
            assets = df_value["total assets"].values

            save_dict = OrderedDict({
                "Profit Margin": tr * 100,
                "Excess Profit": tr * 100,
                "daily_return": daily_return,
                "total_assets": assets
            })

            if self.task == 'test_dynamic':
                metric_save_path = osp.join(self.work_dir,
                                            f'metric_{self.task}_{self.test_dynamic}_{self.test_id}_{self.task_index}.pickle')
                with open(metric_save_path, 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return self.state, self.reward, terminated, truncated, {
                "sharpe_ratio": sharpe_ratio,
                "total_assets": assets,
                "table": table
            }

        weights = self.softmax(actions)
        self.weights_memory.append(weights)
        last_day_memory = self.data

        self.day += 1
        self.data = self.df.loc[self.day, :]

        tic_list = list(self.data.tic)
        s_market = np.array([
            self.data[tech].values.tolist()
            for tech in self.tech_indicator_list
        ]).reshape(-1).astype(np.float32).tolist()

        X = []
        for tic in tic_list:
            df_tic = self.df[self.df.tic == tic]
            df_information = df_tic[self.day - self.length_day:self.day][self.tech_indicator_list].to_numpy()
            df_information = torch.from_numpy(df_information).float().unsqueeze(0)
            X.append(df_information)

        X = torch.cat(X, dim=0).unsqueeze(0).to("cpu")
        y = self.net(X).cpu().detach().squeeze().numpy().astype(np.float32).tolist()

        self.state = np.array(s_market + y, dtype=np.float32)

        portfolio_weights = weights[1:]
        portfolio_return = sum(
            ((self.data.close.values / last_day_memory.close.values) - 1) * portfolio_weights
        )

        weights_brandnew = self.normalization(
            [weights[0]] + list(np.array(weights[1:]) * (self.data.close.values / last_day_memory.close.values))
        )

        self.weights_memory.append(weights_brandnew)

        weights_old = self.weights_memory[-3]
        weights_new = self.weights_memory[-2]
        diff_weights = np.sum(np.abs(np.array(weights_old) - np.array(weights_new)))
        transaction_fee = diff_weights * self.transaction_cost_pct * self.portfolio_value

        new_portfolio_value = (self.portfolio_value - transaction_fee) * (1 + portfolio_return)
        portfolio_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value

        self.reward = np.float32(new_portfolio_value - self.portfolio_value)
        self.portfolio_value = new_portfolio_value

        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.data.date.unique()[0])
        self.asset_memory.append(new_portfolio_value)

        return self.state, self.reward, terminated, truncated, {"weights_brandnew": weights_brandnew}

    def normalization(self, actions):
        actions = np.array(actions)
        return actions / (np.sum(actions) + 1e-10)

    def softmax(self, actions):
        e = np.exp(actions - np.max(actions))  # numerical stability
        return e / (np.sum(e) + 1e-10)

    def save_portfolio_return_memory(self):
        df_return = pd.DataFrame({
            "date": self.date_memory,
            "daily_return": self.portfolio_return_memory
        }).set_index("date")
        return df_return

    def save_asset_memory(self):
        df_value = pd.DataFrame({
            "date": self.date_memory,
            "total assets": self.asset_memory
        }).set_index("date")
        return df_value

    def analysis_result(self):
        df_return = self.save_portfolio_return_memory()
        df_value = self.save_asset_memory()
        df = pd.DataFrame({
            "daily_return": df_return.daily_return.values,
            "total assets": df_value["total assets"].values
        })
        return self.evaluate(df)

    def get_daily_return_rate(self, price_list):
        return [(price_list[i + 1] / price_list[i]) - 1 for i in range(len(price_list) - 1)]

    def evaluate(self, df):
        daily_return = df["daily_return"]
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        tr = df["total assets"].values[-1] / (df["total assets"].values[0] + 1e-10) - 1

        return_rate_list = self.get_daily_return_rate(df["total assets"].values)
        return_rate_list = np.array(return_rate_list, dtype=np.float32)

        sharpe_ratio = np.mean(return_rate_list) * np.sqrt(252) / (np.std(return_rate_list) + 1e-10)
        vol = np.std(return_rate_list)

        peak = df["total assets"][0]
        mdd = 0
        for value in df["total assets"]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            mdd = max(mdd, dd)

        cr = np.sum(daily_return) / (mdd + 1e-10)
        sor = np.sum(daily_return) / (np.std(neg_ret_lst) + 1e-10) / (np.sqrt(len(daily_return)) + 1e-10)
        return tr, sharpe_ratio, vol, mdd, cr, sor
