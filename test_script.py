"""
T1机器人测试和可视化程序
加载训练好的模型进行测试，支持键盘控制
"""

import numpy as np
import mujoco
import mujoco.viewer
import argparse
import time

from ppo_algorithm import PPOAgent
from t1_robot_env import T1RobotEnv


class KeyboardController:
    """键盘控制器"""
    
    def __init__(self):
        """初始化控制器"""
        self.command_vx = 0.0
        self.command_vy = 0.0
        self.command_vyaw = 0.0
        
        # 速度增量
        self.linear_speed_increment = 0.1
        self.angular_speed_increment = 0.2
        
        # 最大速度
        self.max_linear_speed = 2.0
        self.max_angular_speed = 2.0
    
    def update_from_key(self, key: str):
        """
        根据按键更新指令
        
        键位说明:
        - W/S: 前进/后退
        - A/D: 左转/右转
        - Q/E: 向左/向右平移
        - Space: 停止所有运动
        - R: 重置指令
        """
        if key == 'W' or key == 'w':
            self.command_vx = min(self.command_vx + self.linear_speed_increment, self.max_linear_speed)
        elif key == 'S' or key == 's':
            self.command_vx = max(self.command_vx - self.linear_speed_increment, -self.max_linear_speed)
        elif key == 'A' or key == 'a':
            self.command_vyaw = min(self.command_vyaw + self.angular_speed_increment, self.max_angular_speed)
        elif key == 'D' or key == 'd':
            self.command_vyaw = max(self.command_vyaw - self.angular_speed_increment, -self.max_angular_speed)
        elif key == 'Q' or key == 'q':
            self.command_vy = min(self.command_vy + self.linear_speed_increment, self.max_linear_speed)
        elif key == 'E' or key == 'e':
            self.command_vy = max(self.command_vy - self.linear_speed_increment, -self.max_linear_speed)
        elif key == ' ':  # Space键停止
            self.command_vx = 0.0
            self.command_vy = 0.0
            self.command_vyaw = 0.0
        elif key == 'R' or key == 'r':
            self.reset()
    
    def reset(self):
        """重置所有指令"""
        self.command_vx = 0.0
        self.command_vy = 0.0
        self.command_vyaw = 0.0
    
    def get_command(self):
        """获取当前指令"""
        return self.command_vx, self.command_vy, self.command_vyaw
    
    def print_status(self):
        """打印当前状态"""
        print(f"\r指令 - 前进: {self.command_vx:.2f} m/s | "
              f"侧移: {self.command_vy:.2f} m/s | "
              f"旋转: {self.command_vyaw:.2f} rad/s", end='')


def test_policy(
    xml_path: str,
    model_path: str,
    n_episodes: int = 10,
    render: bool = True,
    keyboard_control: bool = True
):
    """
    测试训练好的策略
    
    Args:
        xml_path: MuJoCo模型XML文件路径
        model_path: 训练好的模型路径
        n_episodes: 测试回合数
        render: 是否渲染可视化
        keyboard_control: 是否启用键盘控制
    """
    # 创建环境
    print("正在加载环境...")
    render_mode = 'human' if render else None
    env = T1RobotEnv(xml_path, render_mode=render_mode)
    
    # 创建智能体并加载模型
    print("正在加载模型...")
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim
    )
    agent.load(model_path)
    
    # 键盘控制器
    controller = KeyboardController() if keyboard_control else None
    
    # 如果有渲染，创建viewer
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
    
    print("\n测试开始!")
    if keyboard_control:
        print("\n键位说明:")
        print("  W/S: 前进/后退")
        print("  A/D: 左转/右转")
        print("  Q/E: 向左/向右平移")
        print("  Space: 停止所有运动")
        print("  R: 重置指令")
        print("  ESC: 退出程序")
        print("-" * 60)
    
    # 测试循环
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # 随机初始指令（如果不用键盘控制）
        if not keyboard_control:
            vx = np.random.uniform(0.0, 1.5)
            vy = np.random.uniform(-0.3, 0.3)
            vyaw = np.random.uniform(-0.5, 0.5)
            env.set_command(vx, vy, vyaw)
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done:
            # 键盘控制
            if keyboard_control and controller:
                # 这里简化处理，实际应用中需要事件循环
                # 更新环境指令
                vx, vy, vyaw = controller.get_command()
                env.set_command(vx, vy, vyaw)
                controller.print_status()
            
            # 获取动作（确定性策略）
            action, _, _ = agent.get_action(obs, deterministic=True)
            
            # 环境步进
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # 渲染
            if render and viewer:
                viewer.sync()
                time.sleep(0.01)  # 控制帧率
            
            # 检查最大步数
            if episode_length >= env.max_episode_steps:
                done = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"\nEpisode {episode + 1} 完成:")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  步数: {episode_length}")
        print("-" * 60)
    
    # 输出统计信息
    print("\n测试完成!")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    # 关闭
    if viewer:
        viewer.close()
    env.close()


def record_video(
    xml_path: str,
    model_path: str,
    save_path: str = './demo_video.mp4',
    duration: int = 30,
    fps: int = 30
):
    """
    录制演示视频
    
    Args:
        xml_path: MuJoCo模型XML文件路径
        model_path: 训练好的模型路径
        save_path: 视频保存路径
        duration: 视频时长（秒）
        fps: 帧率
    """
    import cv2
    
    # 创建环境
    print("正在准备录制...")
    env = T1RobotEnv(xml_path)
    
    # 创建智能体并加载模型
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim
    )
    agent.load(model_path)
    
    # 创建渲染器
    renderer = mujoco.Renderer(env.model, height=720, width=1280)
    
    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (1280, 720))
    
    # 开始录制
    obs = env.reset()
    frames_to_record = duration * fps
    
    print(f"开始录制 {duration} 秒视频...")
    
    for frame_idx in range(frames_to_record):
        # 获取动作
        action, _, _ = agent.get_action(obs, deterministic=True)
        
        # 环境步进
        obs, reward, done, info = env.step(action)
        
        # 如果episode结束，重置
        if done:
            obs = env.reset()
        
        # 渲染帧
        renderer.update_scene(env.data)
        pixels = renderer.render()
        
        # BGR转换（OpenCV使用BGR）
        frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        
        # 进度显示
        if (frame_idx + 1) % fps == 0:
            print(f"已录制 {(frame_idx + 1) // fps} 秒...")
    
    # 释放资源
    video_writer.release()
    env.close()
    
    print(f"视频已保存到: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试T1机器人策略')
    parser.add_argument('--xml_path', type=str, required=True,
                        help='MuJoCo模型XML文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'record'],
                        help='运行模式: test(测试) 或 record(录制视频)')
    parser.add_argument('--n_episodes', type=int, default=5,
                        help='测试回合数')
    parser.add_argument('--no_render', action='store_true',
                        help='禁用渲染')
    parser.add_argument('--no_keyboard', action='store_true',
                        help='禁用键盘控制')
    parser.add_argument('--video_path', type=str, default='./demo_video.mp4',
                        help='视频保存路径')
    parser.add_argument('--duration', type=int, default=30,
                        help='视频时长（秒）')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_policy(
            xml_path=args.xml_path,
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            render=not args.no_render,
            keyboard_control=not args.no_keyboard
        )
    elif args.mode == 'record':
        record_video(
            xml_path=args.xml_path,
            model_path=args.model_path,
            save_path=args.video_path,
            duration=args.duration
        )


if __name__ == '__main__':
    # 示例运行
    test_policy(
        xml_path='./models/t1_robot.xml',
        model_path='./checkpoints/final_model.pt',
        n_episodes=3,
        render=True,
        keyboard_control=True
    )
