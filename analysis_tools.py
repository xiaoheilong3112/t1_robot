"""
训练结果分析和可视化工具
用于分析训练日志、绘制学习曲线等
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import argparse


def load_training_log(log_path: str) -> List[Dict]:
    """
    加载训练日志
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        日志条目列表
    """
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    return log_data


def plot_learning_curves(log_data: List[Dict], save_path: str = None):
    """
    绘制学习曲线
    
    Args:
        log_data: 训练日志数据
        save_path: 图片保存路径（可选）
    """
    # 提取数据
    timesteps = [entry['timestep'] for entry in log_data]
    rewards = [entry['mean_episode_reward'] for entry in log_data]
    lengths = [entry['mean_episode_length'] for entry in log_data]
    policy_losses = [entry['policy_loss'] for entry in log_data]
    value_losses = [entry['value_loss'] for entry in log_data]
    entropies = [entry['entropy'] for entry in log_data]
    
    # 创建子图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('T1机器人训练学习曲线', fontsize=16, fontproperties='SimHei')
    
    # 1. 平均回合奖励
    axes[0, 0].plot(timesteps, rewards, linewidth=2, color='blue')
    axes[0, 0].set_xlabel('训练步数', fontproperties='SimHei')
    axes[0, 0].set_ylabel('平均回合奖励', fontproperties='SimHei')
    axes[0, 0].set_title('回合奖励变化', fontproperties='SimHei')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 平均回合长度
    axes[0, 1].plot(timesteps, lengths, linewidth=2, color='green')
    axes[0, 1].set_xlabel('训练步数', fontproperties='SimHei')
    axes[0, 1].set_ylabel('平均回合长度', fontproperties='SimHei')
    axes[0, 1].set_title('回合长度变化', fontproperties='SimHei')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 策略损失
    axes[1, 0].plot(timesteps, policy_losses, linewidth=2, color='red')
    axes[1, 0].set_xlabel('训练步数', fontproperties='SimHei')
    axes[1, 0].set_ylabel('策略损失', fontproperties='SimHei')
    axes[1, 0].set_title('策略网络损失', fontproperties='SimHei')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 价值损失
    axes[1, 1].plot(timesteps, value_losses, linewidth=2, color='orange')
    axes[1, 1].set_xlabel('训练步数', fontproperties='SimHei')
    axes[1, 1].set_ylabel('价值损失', fontproperties='SimHei')
    axes[1, 1].set_title('价值网络损失', fontproperties='SimHei')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 熵
    axes[2, 0].plot(timesteps, entropies, linewidth=2, color='purple')
    axes[2, 0].set_xlabel('训练步数', fontproperties='SimHei')
    axes[2, 0].set_ylabel('策略熵', fontproperties='SimHei')
    axes[2, 0].set_title('策略探索性（熵）', fontproperties='SimHei')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. 奖励平滑曲线（移动平均）
    window_size = min(50, len(rewards) // 10)
    if window_size > 1:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_timesteps = timesteps[window_size-1:]
        axes[2, 1].plot(timesteps, rewards, alpha=0.3, color='blue', label='原始数据')
        axes[2, 1].plot(smoothed_timesteps, smoothed_rewards, linewidth=2, color='blue', label='平滑曲线')
        axes[2, 1].set_xlabel('训练步数', fontproperties='SimHei')
        axes[2, 1].set_ylabel('平均回合奖励', fontproperties='SimHei')
        axes[2, 1].set_title('奖励平滑曲线', fontproperties='SimHei')
        axes[2, 1].legend(prop={'family': 'SimHei'})
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习曲线已保存到: {save_path}")
    else:
        plt.show()


def print_training_summary(log_data: List[Dict]):
    """
    打印训练摘要统计
    
    Args:
        log_data: 训练日志数据
    """
    if not log_data:
        print("日志数据为空")
        return
    
    # 提取数据
    rewards = [entry['mean_episode_reward'] for entry in log_data]
    lengths = [entry['mean_episode_length'] for entry in log_data]
    
    # 计算统计量
    final_reward = rewards[-1]
    max_reward = max(rewards)
    mean_reward = np.mean(rewards)
    
    final_length = lengths[-1]
    max_length = max(lengths)
    mean_length = np.mean(lengths)
    
    total_timesteps = log_data[-1]['timestep']
    total_updates = log_data[-1]['n_updates']
    
    # 计算改进率
    initial_reward = rewards[0] if rewards[0] != 0 else 1.0
    improvement = ((final_reward - initial_reward) / abs(initial_reward)) * 100
    
    print("\n" + "="*60)
    print("训练摘要统计")
    print("="*60)
    print(f"\n训练进度:")
    print(f"  总训练步数: {total_timesteps:,}")
    print(f"  总更新次数: {total_updates:,}")
    
    print(f"\n回合奖励:")
    print(f"  最终奖励: {final_reward:.2f}")
    print(f"  最高奖励: {max_reward:.2f}")
    print(f"  平均奖励: {mean_reward:.2f}")
    print(f"  改进率: {improvement:+.1f}%")
    
    print(f"\n回合长度:")
    print(f"  最终长度: {final_length:.1f}")
    print(f"  最长长度: {max_length:.1f}")
    print(f"  平均长度: {mean_length:.1f}")
    
    # 最近表现
    recent_n = min(10, len(rewards))
    recent_rewards = rewards[-recent_n:]
    recent_mean = np.mean(recent_rewards)
    recent_std = np.std(recent_rewards)
    
    print(f"\n最近{recent_n}次更新:")
    print(f"  平均奖励: {recent_mean:.2f} ± {recent_std:.2f}")
    print("="*60 + "\n")


def compare_models(log_paths: List[str], labels: List[str] = None):
    """
    比较多个训练运行的结果
    
    Args:
        log_paths: 日志文件路径列表
        labels: 每个运行的标签
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(log_paths))]
    
    plt.figure(figsize=(12, 6))
    
    for log_path, label in zip(log_paths, labels):
        log_data = load_training_log(log_path)
        timesteps = [entry['timestep'] for entry in log_data]
        rewards = [entry['mean_episode_reward'] for entry in log_data]
        
        # 平滑曲线
        window_size = min(20, len(rewards) // 10)
        if window_size > 1:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            smoothed_timesteps = timesteps[window_size-1:]
            plt.plot(smoothed_timesteps, smoothed_rewards, linewidth=2, label=label)
        else:
            plt.plot(timesteps, rewards, linewidth=2, label=label)
    
    plt.xlabel('训练步数', fontproperties='SimHei', fontsize=12)
    plt.ylabel('平均回合奖励', fontproperties='SimHei', fontsize=12)
    plt.title('不同训练运行对比', fontproperties='SimHei', fontsize=14)
    plt.legend(prop={'family': 'SimHei'})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_reward_components(log_data: List[Dict]):
    """
    分析奖励函数各组成部分的贡献
    注意：需要在训练时记录详细的奖励分解信息
    
    Args:
        log_data: 训练日志数据
    """
    # 这里假设日志中包含奖励分解信息
    # 实际使用时需要修改训练代码来记录这些信息
    print("奖励组成分析功能需要在训练时记录详细的奖励分解信息")
    print("请修改训练代码，在日志中添加各项奖励的详细值")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析T1机器人训练结果')
    parser.add_argument('--log_path', type=str, required=True,
                        help='训练日志文件路径')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='保存学习曲线图的路径')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='比较多个训练运行（提供多个日志路径）')
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较模式
        print("比较多个训练运行...")
        compare_models(args.compare)
    else:
        # 单个分析
        print(f"加载训练日志: {args.log_path}")
        log_data = load_training_log(args.log_path)
        
        # 打印摘要
        print_training_summary(log_data)
        
        # 绘制学习曲线
        plot_learning_curves(log_data, args.save_plot)


if __name__ == '__main__':
    # 示例使用
    # python analysis_tools.py --log_path ./logs/training_log.json --save_plot ./learning_curves.png
    
    # 或直接运行
    log_path = './logs/final_training_log.json'
    if Path(log_path).exists():
        log_data = load_training_log(log_path)
        print_training_summary(log_data)
        plot_learning_curves(log_data, './learning_curves.png')
    else:
        print(f"日志文件不存在: {log_path}")
        print("请先运行训练程序生成日志")
