"""
T1机器人训练主程序
运行此脚本开始训练
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np

# 导入环境和算法（需要将上面的代码保存为对应的模块）
from ppo_algorithm import PPOAgent
from t1_robot_env import T1RobotEnv


class RolloutBuffer:
    """经验回放缓冲区"""

    def __init__(self):
        """初始化缓冲区"""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        """添加一步经验"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self):
        """获取所有经验"""
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.log_probs),
            self.rewards,
            self.values,
            self.dones,
        )

    def clear(self):
        """清空缓冲区"""
        self.__init__()

    def __len__(self):
        """返回缓冲区大小"""
        return len(self.observations)


def train(
    xml_path: str,
    total_timesteps: int = 10_000_000,
    n_steps: int = 2048,
    save_freq: int = 100,
    log_freq: int = 10,
    save_dir: str = "./checkpoints",
    log_dir: str = "./logs",
):
    """
    训练函数

    Args:
        xml_path: MuJoCo模型XML文件路径
        total_timesteps: 总训练步数
        n_steps: 每次更新收集的步数
        save_freq: 保存频率（迭代次数）
        log_freq: 日志记录频率（迭代次数）
        save_dir: 模型保存目录
        log_dir: 日志保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 创建环境
    print("正在初始化环境...")
    env = T1RobotEnv(xml_path)

    # 创建智能体
    print("正在初始化智能体...")
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    # 训练统计
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    # 初始化
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    n_updates = 0

    # 创建缓冲区
    buffer = RolloutBuffer()

    # 训练日志
    train_log = []

    print(f"\n开始训练，目标步数: {total_timesteps}")
    print("=" * 60)

    start_time = time.time()

    # 主训练循环
    for timestep in range(1, total_timesteps + 1):
        # 获取动作
        action, log_prob, value = agent.get_action(obs)

        # 环境步进
        next_obs, reward, done, info = env.step(action)

        # 存储经验
        buffer.add(obs, action, log_prob, reward, value, done)

        # 更新统计
        episode_reward += reward
        episode_length += 1

        obs = next_obs

        # Episode结束
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # 重置环境
            obs = env.reset()
            episode_reward = 0
            episode_length = 0

        # 当收集足够步数时进行更新
        if len(buffer) >= n_steps:
            # 计算最后一个状态的价值
            with torch.no_grad():
                _, _, next_value = agent.get_action(obs)

            # 获取缓冲区数据
            obs_batch, action_batch, log_prob_batch, rewards, values, dones = (
                buffer.get()
            )

            # 计算GAE
            advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

            # 更新策略
            train_stats = agent.update(
                obs_batch,
                action_batch,
                log_prob_batch,
                advantages,
                returns,
                n_epochs=10,
                batch_size=64,
            )

            # 清空缓冲区
            buffer.clear()
            n_updates += 1

            # 记录训练信息
            if n_updates % log_freq == 0:
                elapsed_time = time.time() - start_time
                fps = timestep / elapsed_time

                mean_reward = np.mean(episode_rewards) if episode_rewards else 0
                mean_length = np.mean(episode_lengths) if episode_lengths else 0

                log_entry = {
                    "timestep": timestep,
                    "n_updates": n_updates,
                    "mean_episode_reward": mean_reward,
                    "mean_episode_length": mean_length,
                    "fps": fps,
                    **train_stats,
                }

                train_log.append(log_entry)

                print(f"\n步数: {timestep}/{total_timesteps} | 更新: {n_updates}")
                print(f"平均奖励: {mean_reward:.2f} | 平均长度: {mean_length:.1f}")
                print(
                    f"策略损失: {train_stats['policy_loss']:.4f} | "
                    f"价值损失: {train_stats['value_loss']:.4f} | "
                    f"熵: {train_stats['entropy']:.4f}"
                )
                print(f"FPS: {fps:.1f}")
                print("-" * 60)

            # 保存模型
            if n_updates % save_freq == 0:
                save_path = os.path.join(save_dir, f"model_update_{n_updates}.pt")
                agent.save(save_path)

                # 保存训练日志
                log_path = os.path.join(log_dir, "training_log.json")
                with open(log_path, "w") as f:
                    json.dump(train_log, f, indent=2)

    # 训练结束，保存最终模型
    final_save_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_save_path)

    # 保存最终日志
    final_log_path = os.path.join(log_dir, "final_training_log.json")
    with open(final_log_path, "w") as f:
        json.dump(train_log, f, indent=2)

    print("\n训练完成!")
    print(f"总用时: {(time.time() - start_time) / 3600:.2f} 小时")
    print(f"模型已保存到: {final_save_path}")

    env.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练T1机器人行走策略")
    parser.add_argument(
        "--xml_path", type=str, required=True, help="MuJoCo模型XML文件路径"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=10_000_000, help="总训练步数"
    )
    parser.add_argument("--n_steps", type=int, default=2048, help="每次更新收集的步数")
    parser.add_argument("--save_freq", type=int, default=100, help="模型保存频率")
    parser.add_argument("--log_freq", type=int, default=10, help="日志记录频率")
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="模型保存目录"
    )
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志保存目录")

    args = parser.parse_args()

    # 开始训练
    train(
        xml_path=args.xml_path,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    # 如果直接运行，需要修改xml_path
    import torch

    train(
        xml_path="./models/t1_robot.xml",  # 修改为实际路径
        total_timesteps=50000,
        n_steps=1024,
        save_freq=20,
        log_freq=5,
    )
