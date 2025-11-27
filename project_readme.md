# T1机器人强化学习训练项目

## 项目概述

本项目使用MuJoCo物理引擎和PPO强化学习算法，训练加速进化T1机器人的行走能力，实现指令跟踪、转向等功能。

**开发环境**: Windows + Python 3.10+ + UV包管理器

## 环境配置

### 系统要求
- **操作系统**: Windows 10/11
- **Python**: 3.10 或更高版本
- **GPU** (可选): NVIDIA GPU with CUDA 11.8+ (推荐用于训练加速)
- **内存**: 8GB+ RAM
- **磁盘空间**: 5GB+

### 快速开始

#### 1. 安装UV包管理器

**方式A: PowerShell (推荐)**
```powershell
# 以管理员身份运行PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**方式B: 使用pip**
```bash
pip install uv
```

验证安装:
```bash
uv --version
```

#### 2. 自动配置环境

```bash
# 在项目根目录下运行
setup.bat
```

脚本会自动：
- 检测UV和Python环境
- 创建必要的目录结构
- 提供安装选项（CPU/GPU版本）
- 验证依赖安装

#### 3. 手动配置（可选）

如果需要手动配置，请参考以下命令：

```bash
# CPU版本
uv pip install -e .

# GPU版本 (CUDA 11.8)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .

# GPU版本 (CUDA 12.1)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .

# 完整开发环境（包含日志和开发工具）
uv pip install -e .[dev,logging]
```

## 项目结构

```
t1_robot_project/
├── models/
│   └── t1_robot.xml          # MuJoCo机器人模型文件
├── src/                       # 源代码目录
│   ├── t1_robot_env.py       # 强化学习环境定义
│   └── ppo_algorithm.py      # PPO算法实现
├── train.py                   # 训练主程序
├── test.py                    # 测试和可视化程序
├── analysis_tools.py          # 训练结果分析工具
├── checkpoints/               # 模型保存目录
├── logs/                      # 训练日志目录
├── videos/                    # 视频录制目录
├── pyproject.toml             # UV项目配置文件
├── setup.bat                  # Windows环境配置脚本
└── README.md                  # 项目说明文档
```

## 运行说明

### 1. 训练模型

```bash
# 基础训练（使用默认参数）
python train.py --xml_path .\models\t1_robot.xml

# 自定义训练参数
python train.py ^
    --xml_path .\models\t1_robot.xml ^
    --total_timesteps 10000000 ^
    --n_steps 2048 ^
    --save_freq 100 ^
    --log_freq 10 ^
    --save_dir .\checkpoints ^
    --log_dir .\logs
```

**训练参数说明：**
- `--xml_path`: MuJoCo模型文件路径
- `--total_timesteps`: 总训练步数（默认10M）
- `--n_steps`: 每次更新收集的步数（默认2048）
- `--save_freq`: 模型保存频率（默认每100次更新）
- `--log_freq`: 日志记录频率（默认每10次更新）

### 2. 测试模型

```bash
# 可视化测试（支持键盘控制）
python test.py ^
    --xml_path .\models\t1_robot.xml ^
    --model_path .\checkpoints\final_model.pt ^
    --mode test ^
    --n_episodes 5

# 录制演示视频
python test.py ^
    --xml_path .\models\t1_robot.xml ^
    --model_path .\checkpoints\final_model.pt ^
    --mode record ^
    --video_path .\videos\demo.mp4 ^
    --duration 30
```

### 3. 分析训练结果

```bash
# 生成学习曲线图
python analysis_tools.py ^
    --log_path .\logs\training_log.json ^
    --save_plot .\learning_curves.png

# 比较多个训练运行
python analysis_tools.py ^
    --compare .\logs\run1.json .\logs\run2.json
```

### 4. 键位操作说明

在测试模式下，使用以下键位控制机器人：

| 按键 | 功能 |
|------|------|
| **W** | 增加前进速度 |
| **S** | 增加后退速度 |
| **A** | 向左旋转 |
| **D** | 向右旋转 |
| **Q** | 向左平移 |
| **E** | 向右平移 |
| **Space** | 停止所有运动 |
| **R** | 重置所有指令 |
| **ESC** | 退出程序 |

## 架构设计说明

### 1. 环境设计（T1RobotEnv）

**观察空间：**
- 基座方向向量（3维）：机器人相对于重力的姿态
- 基座角速度（3维）：机器人本体坐标系下的旋转速度
- 关节位置（n维）：所有关节的当前角度
- 关节速度（n维）：所有关节的角速度
- 上一步动作（n维）：用于动作平滑
- 指令速度（3维）：期望的前进、侧移和旋转速度

**动作空间：**
- 连续动作，取值范围 [-1, 1]
- 每个关节对应一个动作维度
- 通过执行器（电机）施加到关节上

**状态转换：**
```
观察 → 策略网络 → 动作 → MuJoCo仿真 → 新观察 + 奖励
```

### 2. PPO算法实现

**核心组件：**

1. **Actor-Critic网络**
   - Actor（策略网络）：输出动作均值
   - Critic（价值网络）：评估状态价值
   - 共享观察编码层

2. **训练流程**
   ```
   收集经验 → 计算GAE优势 → 多轮更新策略 → 重复
   ```

3. **关键技术**
   - GAE（广义优势估计）：平衡偏差和方差
   - PPO裁剪：限制策略更新幅度
   - 价值函数裁剪：稳定训练
   - 梯度裁剪：防止梯度爆炸

### 3. 奖励函数设计

奖励函数是训练成功的关键，设计思路如下：

#### 主要奖励项

1. **速度跟踪奖励**（权重最高）
   ```python
   velocity_reward = -2.0 * (vx_error² + vy_error² + 0.5 * vyaw_error²)
   ```
   - 鼓励机器人按照指令速度运动
   - 分别惩罚前进、侧移和旋转速度的偏差
   - 旋转误差权重相对较小（0.5倍）

2. **姿态稳定奖励**
   ```python
   orientation_reward = -5.0 * (gravity_x² + gravity_y²)
   ```
   - 惩罚机器人倾斜
   - 鼓励保持直立姿态

3. **高度保持奖励**
   ```python
   height_reward = -3.0 * (height - target_height)²
   ```
   - 鼓励保持合适的站立高度
   - 防止蹲下或过度站立

#### 次要奖励项

4. **能量消耗惩罚**
   ```python
   energy_penalty = -0.01 * Σ(action²)
   ```
   - 鼓励高效运动
   - 避免不必要的大幅度动作

5. **动作平滑性奖励**
   ```python
   action_smoothness = -0.1 * Σ((action_t - action_{t-1})²)
   ```
   - 惩罚相邻步骤动作的剧烈变化
   - 使机器人运动更平滑自然

6. **关节速度惩罚**
   ```python
   joint_vel_penalty = -0.01 * Σ(joint_vel²)
   ```
   - 避免关节过快运动
   - 提高动作稳定性

7. **存活奖励**
   ```python
   alive_reward = 1.0
   ```
   - 固定奖励，鼓励长时间保持站立

#### 总奖励
```python
total_reward = velocity_reward + orientation_reward + height_reward +
               energy_penalty + action_smoothness + 
               joint_vel_penalty + alive_reward
```

#### 奖励设计原则

1. **稀疏 vs 密集**：使用密集奖励帮助探索
2. **权重平衡**：主要目标权重大，辅助目标权重小
3. **可调性**：所有权重都可以根据训练效果调整
4. **避免奖励黑客**：多个奖励项相互制约

## 开发过程中的问题及解决方案

### 1. 机器人频繁跌倒

**问题：** 训练初期机器人无法保持平衡，经常跌倒。

**解决方案：**
- 增加姿态稳定奖励的权重（从1.0提升到5.0）
- 调整初始化策略：在reset时添加小的随机扰动
- 降低动作输出幅度：使用tanh激活并进行缩放
- 增加关节阻尼，提高稳定性

### 2. 训练不稳定

**问题：** 训练过程中奖励值震荡剧烈，性能时好时坏。

**解决方案：**
- 使用GAE计算优势值，平衡偏差和方差
- 实施梯度裁剪（max_grad_norm=0.5）
- 增加训练epoch数（从4提升到10）
- 使用Adam优化器并调整学习率（3e-4）

### 3. 指令跟随不准确

**问题：** 机器人倾向于忽略速度指令，始终以固定速度运动。

**解决方案：**
- 提高速度跟踪奖励的权重
- 在每个episode开始时随机采样指令
- 将指令速度加入观察空间
- 实施curriculum learning：先训练简单指令，再增加复杂度

### 4. 仿真性能问题

**问题：** 训练速度慢，仿真FPS低。

**解决方案：**
- 禁用训练时的渲染
- 使用较大的仿真时间步（0.002s）
- 批量环境并行训练（可选，需要vectorized env）
- 使用GPU加速神经网络计算

### 5. 动作抖动

**问题：** 机器人运动不够平滑，存在高频抖动。

**解决方案：**
- 添加动作平滑性惩罚
- 增加关节阻尼参数
- 使用低通滤波器处理动作输出
- 调整PD控制器参数

## 未完成的工作及思路

### 1. 复杂地形适应

**目标：** 让机器人能够翻越楼梯、斜坡等复杂地形。

**实现思路：**
- 修改场景XML，添加楼梯、斜坡等几何体
- 使用课程学习：从平地 → 小斜坡 → 大斜坡 → 楼梯
- 引入高度图传感器，提供前方地形信息
- 调整奖励函数，奖励爬升高度
- 参考论文：Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning

### 2. 抗扰动能力

**目标：** 增强机器人对外部推力的抵抗能力。

**实现思路：**
- 在训练过程中随机施加外力扰动
- 添加扰动相关的观察（如果有力传感器）
- 使用domain randomization：随机化质量、摩擦力等参数
- 奖励函数中考虑恢复速度

### 3. 自动恢复站立

**目标：** 机器人倒地后能够自己站起来。

**实现思路：**
- 修改终止条件，不在倒地时立即结束episode
- 设计分阶段奖励：倒地 → 翻身 → 四肢支撑 → 站立
- 使用hindsight experience replay (HER)
- 可能需要分开训练两个策略：行走策略和起身策略

### 4. 自然语言控制

**目标：** 使用文字或语音指挥机器人运动。

**实现思路：**
```
语音/文字 → 语音识别/NLP → 速度指令 → RL策略 → 动作
```

具体步骤：
1. 构建指令到速度的映射
   - "前进" → vx=1.0, vy=0, vyaw=0
   - "左转" → vx=0.5, vy=0, vyaw=0.5
   - "快跑" → vx=2.0, vy=0, vyaw=0

2. 集成语音识别
   - 使用Whisper等模型进行语音转文字
   - 实时流式处理

3. 使用大语言模型理解复杂指令
   - "绕着桌子走一圈" → 一系列基础运动指令
   - Claude/GPT等作为高层规划器

### 5. 多模态感知

**目标：** 添加视觉、触觉等传感器。

**实现思路：**
- 在MuJoCo中添加摄像头传感器
- 使用CNN提取视觉特征
- 融合本体感知和外部感知
- 端到端训练或分阶段训练

### 6. Sim-to-Real迁移

**目标：** 将仿真训练的策略部署到真实机器人。

**实现思路：**
- Domain randomization：随机化仿真参数
- 系统辨识：精确建模真实机器人
- Residual RL：在真实机器人上微调
- 使用真实世界数据增强训练

## 参考资料清单

### 论文
1. **PPO算法**
   - Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
   - 核心算法实现的理论基础

2. **Legged Robotics**
   - Rudin et al. "Learning to Walk in Minutes Using Massively Parallel Deep RL" (2022)
   - 四足/双足机器人学习的最新方法

3. **奖励设计**
   - Ng et al. "Policy Invariance Under Reward Transformations" (1999)
   - 奖励塑形的理论基础

### 开源项目
1. **IsaacGym / IsaacLab**
   - NVIDIA的并行物理仿真环境
   - https://github.com/isaac-sim/IsaacLab

2. **Legged Gym**
   - ETH Zurich的腿式机器人训练框架
   - https://github.com/leggedrobotics/legged_gym

3. **MuJoCo Menagerie**
   - 官方机器人模型库
   - https://github.com/google-deepmind/mujoco_menagerie

### 文档
1. **MuJoCo Documentation**
   - https://mujoco.readthedocs.io/
   - 物理引擎官方文档

2. **Stable-Baselines3**
   - https://stable-baselines3.readthedocs.io/
   - RL算法实现参考

### 视频教程
1. **Deep RL Course (Hugging Face)**
   - https://huggingface.co/learn/deep-rl-course/
   - 强化学习基础和实践

2. **MuJoCo Tutorial**
   - DeepMind官方教程
   - 仿真环境搭建

## 进阶目标实现建议

### 楼梯攀爬训练
```python
# 在环境中添加楼梯
class StairEnv(T1RobotEnv):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.curriculum_level = 0
        
    def _add_stairs(self, n_steps, height_increment):
        # 动态生成楼梯几何体
        pass
    
    def _compute_reward(self, action):
        # 额外奖励爬升高度
        height_gain_reward = 2.0 * (current_height - initial_height)
        return base_reward + height_gain_reward
```

### 语言指令接口
```python
class LanguageController:
    def __init__(self, llm_model):
        self.llm = llm_model
        
    def parse_command(self, text: str) -> tuple:
        """将自然语言转换为速度指令"""
        prompt = f"将以下指令转换为机器人速度命令：{text}"
        response = self.llm(prompt)
        return self._extract_velocities(response)
```

## 总结

本项目实现了T1机器人的基础行走能力训练，包括：
- ✅ 完整的MuJoCo仿真环境
- ✅ PPO强化学习算法
- ✅ 指令跟随能力
- ✅ 稳定的站立和行走
- ✅ 键盘控制接口
- ✅ 模型保存和加载
- ✅ 可视化测试工具

项目为后续的复杂功能开发提供了坚实的基础架构。通过模块化设计，可以方便地扩展新功能，如复杂地形适应、抗扰动训练、自然语言控制等。

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- GitHub Issues
- Email: [你的邮箱]

## 许可证

本项目基于MIT许可证开源。
