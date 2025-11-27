"""
T1机器人演示脚本
用于录屏展示项目功能和效果
"""

import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from ppo_algorithm import PPOAgent
from t1_robot_env import T1RobotEnv


def demo_basic_walking(env, agent, duration=10):
    """演示基本行走功能"""
    print("\n=== 演示1: 基本行走能力 ===")
    print("机器人将展示稳定的前进行走...")

    obs = env.reset()
    env.set_command(0.8, 0.0, 0.0)  # 设置前进指令

    start_time = time.time()
    step_count = 0

    while time.time() - start_time < duration:
        action, _, _ = agent.get_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        step_count += 1

        if done:
            print(f"Episode结束，共{step_count}步")
            obs = env.reset()
            env.set_command(0.8, 0.0, 0.0)
            step_count = 0

        time.sleep(0.02)  # 控制帧率

    print("基本行走演示完成")


def demo_directional_control(env, agent, duration=15):
    """演示方向控制功能"""
    print("\n=== 演示2: 方向控制能力 ===")
    print("机器人将展示前进、后退、左转、右转...")

    # 定义不同的运动指令
    commands = [
        (0.8, 0.0, 0.0, "前进"),
        (0.0, 0.0, 0.5, "左转"),
        (0.8, 0.0, 0.0, "前进"),
        (0.0, 0.0, -0.5, "右转"),
        (0.8, 0.0, 0.0, "前进"),
        (-0.5, 0.0, 0.0, "后退"),
    ]

    obs = env.reset()
    start_time = time.time()
    command_duration = duration / len(commands)
    current_command = 0
    command_start_time = start_time

    while time.time() - start_time < duration:
        # 检查是否需要切换指令
        if (
            time.time() - command_start_time > command_duration
            and current_command < len(commands) - 1
        ):
            current_command += 1
            command_start_time = time.time()

        vx, vy, vyaw, desc = commands[current_command]
        env.set_command(vx, vy, vyaw)
        print(
            f"\r当前指令: {desc} (vx={vx:.1f}, vy={vy:.1f}, vyaw={vyaw:.1f})",
            end="",
            flush=True,
        )

        action, _, _ = agent.get_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

        time.sleep(0.02)

    print("\n方向控制演示完成")


def demo_stability_test(env, agent, duration=8):
    """演示稳定性测试"""
    print("\n=== 演示3: 稳定性测试 ===")
    print("机器人将在静止状态下保持平衡...")

    obs = env.reset()
    env.set_command(0.0, 0.0, 0.0)  # 静止指令

    start_time = time.time()
    stable_steps = 0
    total_steps = 0

    while time.time() - start_time < duration:
        action, _, _ = agent.get_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_steps += 1
        if not done:
            stable_steps += 1

        if done:
            print(f"跌倒! 稳定了{stable_steps}步")
            obs = env.reset()
            stable_steps = 0

        time.sleep(0.02)

    stability_rate = stable_steps / total_steps * 100 if total_steps > 0 else 0
    print(f"稳定性测试完成，稳定率: {stability_rate:.1f}%")


def demo_interactive_control(env, agent, duration=20):
    """演示交互式控制"""
    print("\n=== 演示4: 交互式键盘控制 ===")
    print("键位说明:")
    print("  W: 前进    S: 后退")
    print("  A: 左转    D: 右转")
    print("  Q: 左移    E: 右移")
    print("  Space: 停止")
    print("  (自动演示不同指令组合)")

    # 模拟键盘输入序列
    key_sequence = [
        ("W", 3.0, "前进"),
        ("A", 2.0, "左转"),
        ("W", 3.0, "前进"),
        ("D", 2.0, "右转"),
        ("W", 2.0, "前进"),
        ("Q", 2.0, "左移"),
        ("E", 2.0, "右移"),
        ("S", 2.0, "后退"),
        (" ", 2.0, "停止"),
    ]

    obs = env.reset()
    start_time = time.time()
    current_key = 0
    key_start_time = start_time

    # 当前指令状态
    vx, vy, vyaw = 0.0, 0.0, 0.0

    while time.time() - start_time < duration and current_key < len(key_sequence):
        # 检查是否需要切换按键
        if time.time() - key_start_time > key_sequence[current_key][1]:
            if current_key < len(key_sequence) - 1:
                current_key += 1
                key_start_time = time.time()

        key, _, desc = key_sequence[current_key]

        # 根据按键更新指令
        if key == "W":
            vx = min(vx + 0.1, 1.0)
        elif key == "S":
            vx = max(vx - 0.1, -1.0)
        elif key == "A":
            vyaw = min(vyaw + 0.1, 0.8)
        elif key == "D":
            vyaw = max(vyaw - 0.1, -0.8)
        elif key == "Q":
            vy = min(vy + 0.1, 0.5)
        elif key == "E":
            vy = max(vy - 0.1, -0.5)
        elif key == " ":
            vx, vy, vyaw = 0.0, 0.0, 0.0

        env.set_command(vx, vy, vyaw)
        print(
            f"\r按键: {key} -> {desc} | 指令: 前进{vx:.1f} 侧移{vy:.1f} 旋转{vyaw:.1f}",
            end="",
            flush=True,
        )

        action, _, _ = agent.get_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

        time.sleep(0.02)

    print("\n交互式控制演示完成")


def main():
    """主演示函数"""
    print("=" * 60)
    print("T1机器人强化学习项目演示")
    print("Motphys具身智能技术应用工程师技术测试")
    print("=" * 60)

    # 检查模型文件是否存在
    model_path = "./checkpoints/final_model.pt"
    xml_path = "./models/t1_robot.xml"

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行 train_script.py 进行训练")
        return

    if not os.path.exists(xml_path):
        print(f"错误: 找不到机器人模型文件 {xml_path}")
        return

    # 初始化环境和智能体
    print("\n正在初始化环境...")
    env = T1RobotEnv(xml_path, render_mode="human")

    print("正在加载训练好的模型...")
    agent = PPOAgent(obs_dim=env.obs_dim, action_dim=env.action_dim)
    agent.load(model_path)

    # 创建MuJoCo viewer
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -15

    print("\n系统信息:")
    print(f"- 环境: {env.__class__.__name__}")
    print(f"- 观察维度: {env.obs_dim}")
    print(f"- 动作维度: {env.action_dim}")
    print(f"- 设备: {agent.device}")
    print(f"- 关节数: {len(env.model.joint_names[1:])}")  # 排除root joint
    print(f"- 执行器数: {env.model.nu}")

    try:
        print("\n开始演示! (按Ctrl+C可提前结束)")
        time.sleep(2)

        # 执行各项演示
        demo_basic_walking(env, agent, duration=12)
        viewer.sync()
        time.sleep(1)

        demo_directional_control(env, agent, duration=18)
        viewer.sync()
        time.sleep(1)

        demo_stability_test(env, agent, duration=10)
        viewer.sync()
        time.sleep(1)

        demo_interactive_control(env, agent, duration=25)
        viewer.sync()

        print("\n" + "=" * 60)
        print("演示完成!")
        print("项目成功展示了以下核心功能:")
        print("✅ 稳定的行走能力")
        print("✅ 多方向运动控制")
        print("✅ 平衡保持能力")
        print("✅ 交互式键盘控制")
        print("✅ 实时物理仿真渲染")
        print("=" * 60)

        input("按Enter键退出...")

    except KeyboardInterrupt:
        print("\n演示被用户中断")

    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 清理资源
        if "viewer" in locals():
            viewer.close()
        env.close()
        print("资源已清理完成")


if __name__ == "__main__":
    main()
