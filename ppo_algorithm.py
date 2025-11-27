"""
PPO (Proximal Policy Optimization) 算法实现
用于训练T1机器人的行走策略
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Actor-Critic神经网络模型"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        初始化网络

        Args:
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(ActorCritic, self).__init__()

        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 动作的对数标准差（可学习参数）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观察值张量

        Returns:
            动作均值和状态价值
        """
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据观察值采样动作

        Args:
            obs: 观察值
            deterministic: 是否使用确定性策略（测试时使用）

        Returns:
            动作、动作对数概率、状态价值
        """
        action_mean, value = self.forward(obs)

        if deterministic:
            return action_mean, None, value

        # 构建正态分布
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        # 采样动作
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的对数概率和熵

        Args:
            obs: 观察值
            actions: 动作

        Returns:
            动作对数概率、状态价值、熵
        """
        action_mean, value = self.forward(obs)

        # 构建正态分布
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        # 计算对数概率和熵
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """PPO强化学习智能体"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化PPO智能体

        Args:
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE的lambda参数
            clip_epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            device: 计算设备
        """
        self.device = torch.device(device)

        # 创建网络
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        print(f"PPO智能体初始化完成，使用设备: {self.device}")

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        获取动作

        Args:
            obs: 观察值
            deterministic: 是否使用确定性策略

        Returns:
            动作、对数概率、价值
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)

            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().numpy() if log_prob is not None else None
            value = value.cpu().numpy()[0]

        return action, log_prob, value

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算广义优势估计(GAE)

        Args:
            rewards: 奖励列表
            values: 价值估计列表
            dones: 终止标志列表
            next_value: 下一个状态的价值

        Returns:
            优势值和回报值
        """
        advantages = []
        returns = []
        gae = 0

        # 从后向前计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            # TD误差
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]

            # GAE累积
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

            # 计算回报
            returns.insert(0, gae + values[t])

        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        return advantages, returns

    def update(
        self,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
        old_log_prob_batch: np.ndarray,
        advantage_batch: np.ndarray,
        return_batch: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """
        使用收集的经验更新策略

        Args:
            obs_batch: 观察批次
            action_batch: 动作批次
            old_log_prob_batch: 旧的动作对数概率
            advantage_batch: 优势值
            return_batch: 回报值
            n_epochs: 训练轮数
            batch_size: 小批次大小

        Returns:
            训练统计信息
        """
        # 标准化优势值
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (
            advantage_batch.std() + 1e-8
        )

        # 转换为张量
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        action_tensor = torch.FloatTensor(action_batch).to(self.device)
        old_log_prob_tensor = torch.FloatTensor(old_log_prob_batch).to(self.device)
        advantage_tensor = torch.FloatTensor(advantage_batch).to(self.device)
        return_tensor = torch.FloatTensor(return_batch).to(self.device)

        # 训练统计
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # 多轮训练
        for epoch in range(n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(obs_batch))

            # 小批次训练
            for start in range(0, len(obs_batch), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # 获取小批次数据
                obs_b = obs_tensor[batch_indices]
                action_b = action_tensor[batch_indices]
                old_log_prob_b = old_log_prob_tensor[batch_indices]
                advantage_b = advantage_tensor[batch_indices]
                return_b = return_tensor[batch_indices]

                # 评估动作
                log_prob, value, entropy = self.policy.evaluate_actions(obs_b, action_b)

                # 计算比率
                ratio = torch.exp(log_prob - old_log_prob_b)

                # PPO裁剪目标
                surr1 = ratio * advantage_b
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * advantage_b
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = 0.5 * (return_b - value).pow(2).mean()

                # 熵奖励
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 记录统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # 返回平均统计
        stats = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

        return stats

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"模型已保存到: {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"模型已从 {path} 加载")
