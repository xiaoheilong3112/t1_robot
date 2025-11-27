"""
T1机器人MuJoCo强化学习训练环境
用于训练机器人的行走、转向和指令跟随能力
"""

import numpy as np
import mujoco
from typing import Tuple, Dict, Optional
import os


class T1RobotEnv:
    """T1机器人仿真环境类"""
    
    def __init__(self, xml_path: str, render_mode: Optional[str] = None):
        """
        初始化环境
        
        Args:
            xml_path: MuJoCo XML模型文件路径
            render_mode: 渲染模式，'human'表示可视化渲染，None表示无渲染
        """
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 渲染设置
        self.render_mode = render_mode
        self.viewer = None
        
        # 环境参数
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # 目标速度指令（前进速度、侧向速度、旋转速度）
        self.command_vx = 0.0  # 前进速度目标 (m/s)
        self.command_vy = 0.0  # 侧向速度目标 (m/s)
        self.command_vyaw = 0.0  # 旋转速度目标 (rad/s)
        
        # 获取关节和执行器信息
        self.n_joints = self.model.njnt
        self.n_actuators = self.model.nu
        
        # 动作空间和观察空间维度
        self.action_dim = self.n_actuators
        self.obs_dim = self._get_obs().shape[0]
        
        # 初始机器人位置
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
        print(f"环境初始化完成: 关节数={self.n_joints}, 执行器数={self.n_actuators}")
        print(f"观察维度={self.obs_dim}, 动作维度={self.action_dim}")
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        重置环境到初始状态
        
        Args:
            seed: 随机种子
            
        Returns:
            初始观察值
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 重置模拟状态
        self.data.qpos[:] = self.init_qpos + np.random.uniform(-0.01, 0.01, self.model.nq)
        self.data.qvel[:] = self.init_qvel + np.random.uniform(-0.01, 0.01, self.model.nv)
        
        # 前向动力学计算
        mujoco.mj_forward(self.model, self.data)
        
        # 重置步数
        self.current_step = 0
        
        # 随机生成新的指令
        self._sample_command()
        
        return self._get_obs()
    
    def _sample_command(self):
        """随机采样运动指令"""
        self.command_vx = np.random.uniform(-1.0, 2.0)  # 前进速度范围
        self.command_vy = np.random.uniform(-0.5, 0.5)  # 侧向速度范围
        self.command_vyaw = np.random.uniform(-1.0, 1.0)  # 旋转速度范围
    
    def set_command(self, vx: float, vy: float, vyaw: float):
        """
        手动设置运动指令
        
        Args:
            vx: 前进速度 (m/s)
            vy: 侧向速度 (m/s)
            vyaw: 旋转速度 (rad/s)
        """
        self.command_vx = vx
        self.command_vy = vy
        self.command_vyaw = vyaw
    
    def _get_obs(self) -> np.ndarray:
        """
        获取当前观察值
        
        Returns:
            观察向量，包含：
            - 基座方向 (3维)
            - 基座角速度 (3维)
            - 关节位置 (n维)
            - 关节速度 (n维)
            - 上一步动作 (n维)
            - 指令速度 (3维)
        """
        # 基座位置和方向
        base_pos = self.data.qpos[:3]
        base_quat = self.data.qpos[3:7]
        
        # 将四元数转换为旋转矩阵，提取重力方向向量
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, base_quat)
        rot_mat = rot_mat.reshape(3, 3)
        gravity_vec = rot_mat.T @ np.array([0, 0, -1])  # 在机器人坐标系中的重力方向
        
        # 基座线速度和角速度（在世界坐标系中）
        base_linvel = self.data.qvel[:3]
        base_angvel = self.data.qvel[3:6]
        
        # 将速度转换到机器人本体坐标系
        base_linvel_local = rot_mat.T @ base_linvel
        base_angvel_local = rot_mat.T @ base_angvel
        
        # 关节位置和速度
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]
        
        # 上一步的动作（初始为0）
        last_action = np.zeros(self.action_dim)
        if hasattr(self, 'last_action'):
            last_action = self.last_action
        
        # 组合观察向量
        obs = np.concatenate([
            gravity_vec,  # 3维
            base_angvel_local,  # 3维
            joint_pos,  # n维
            joint_vel,  # n维
            last_action,  # n维
            [self.command_vx, self.command_vy, self.command_vyaw]  # 3维指令
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步仿真
        
        Args:
            action: 动作向量（关节目标位置或力矩）
            
        Returns:
            observation: 新的观察值
            reward: 奖励值
            done: 是否终止
            info: 额外信息字典
        """
        # 保存当前动作
        self.last_action = action.copy()
        
        # 动作缩放和限幅
        action = np.clip(action, -1.0, 1.0)
        
        # 将动作应用到执行器
        self.data.ctrl[:] = action
        
        # 执行仿真步进（通常需要多个substep保证稳定性）
        for _ in range(5):  # 5个substep对应约0.01秒
            mujoco.mj_step(self.model, self.data)
        
        # 获取新观察
        obs = self._get_obs()
        
        # 计算奖励
        reward, reward_info = self._compute_reward(action)
        
        # 检查终止条件
        done = self._check_termination()
        
        # 更新步数
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True
        
        # 额外信息
        info = {
            'episode_step': self.current_step,
            **reward_info
        }
        
        return obs, reward, done, info
    
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        """
        计算奖励函数
        
        奖励设计思路：
        1. 速度跟踪奖励：鼓励机器人按照指令速度运动
        2. 方向稳定性奖励：惩罚过大的倾斜角度
        3. 能量消耗惩罚：鼓励平滑高效的运动
        4. 动作平滑性奖励：避免抖动
        5. 存活奖励：鼓励保持站立
        
        Args:
            action: 当前动作
            
        Returns:
            总奖励值和各部分奖励的详细信息
        """
        # 获取基座状态
        base_pos = self.data.qpos[:3]
        base_quat = self.data.qpos[3:7]
        base_linvel = self.data.qvel[:3]
        base_angvel = self.data.qvel[3:6]
        
        # 转换到本体坐标系
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, base_quat)
        rot_mat = rot_mat.reshape(3, 3)
        base_linvel_local = rot_mat.T @ base_linvel
        base_angvel_local = rot_mat.T @ base_angvel
        
        # 1. 速度跟踪奖励（主要奖励）
        vx_error = (base_linvel_local[0] - self.command_vx) ** 2
        vy_error = (base_linvel_local[1] - self.command_vy) ** 2
        vyaw_error = (base_angvel_local[2] - self.command_vyaw) ** 2
        
        velocity_reward = -2.0 * (vx_error + vy_error + 0.5 * vyaw_error)
        
        # 2. 方向稳定性奖励（惩罚倾斜）
        # 提取roll和pitch角度
        gravity_vec = rot_mat.T @ np.array([0, 0, -1])
        orientation_reward = -5.0 * (gravity_vec[0]**2 + gravity_vec[1]**2)
        
        # 3. 高度奖励（保持合适高度）
        target_height = 0.5  # 根据机器人设定目标高度
        height_error = (base_pos[2] - target_height) ** 2
        height_reward = -3.0 * height_error
        
        # 4. 能量消耗惩罚
        energy_penalty = -0.01 * np.sum(action ** 2)
        
        # 5. 动作平滑性奖励（与上一步动作的差异）
        if hasattr(self, 'last_action'):
            action_smoothness = -0.1 * np.sum((action - self.last_action) ** 2)
        else:
            action_smoothness = 0.0
        
        # 6. 关节速度惩罚（避免过快运动）
        joint_vel = self.data.qvel[6:]
        joint_vel_penalty = -0.01 * np.sum(joint_vel ** 2)
        
        # 7. 存活奖励
        alive_reward = 1.0
        
        # 总奖励
        total_reward = (
            velocity_reward +
            orientation_reward +
            height_reward +
            energy_penalty +
            action_smoothness +
            joint_vel_penalty +
            alive_reward
        )
        
        # 返回详细信息用于调试
        reward_info = {
            'velocity_reward': velocity_reward,
            'orientation_reward': orientation_reward,
            'height_reward': height_reward,
            'energy_penalty': energy_penalty,
            'action_smoothness': action_smoothness,
            'alive_reward': alive_reward,
            'total_reward': total_reward
        }
        
        return total_reward, reward_info
    
    def _check_termination(self) -> bool:
        """
        检查是否满足终止条件
        
        Returns:
            是否终止episode
        """
        # 获取基座高度
        base_height = self.data.qpos[2]
        
        # 获取基座方向
        base_quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, base_quat)
        rot_mat = rot_mat.reshape(3, 3)
        
        # 计算倾斜角度（z轴在世界坐标系中的方向）
        z_axis = rot_mat[:, 2]
        tilt_angle = np.arccos(np.clip(z_axis[2], -1.0, 1.0))
        
        # 终止条件
        # 1. 高度过低（跌倒）
        if base_height < 0.2:
            return True
        
        # 2. 倾斜角度过大
        if tilt_angle > np.pi / 3:  # 60度
            return True
        
        return False
    
    def render(self):
        """渲染环境（如果启用渲染模式）"""
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
