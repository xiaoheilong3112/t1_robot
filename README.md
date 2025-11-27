# T1机器人强化学习项目

基于PPO算法的T1双足机器人强化学习训练系统，实现机器人稳定行走和多方向控制能力。

## 🎯 项目概述

本项目是Motphys具身智能技术应用工程师技术测试的完整实现，使用MuJoCo物理仿真环境和PPO强化学习算法，训练T1双足机器人具备稳定的行走和控制能力。

### 核心功能

- ✅ **稳定行走**: 机器人能保持11.7步平均稳定行走
- ✅ **多方向控制**: 支持前进、后退、左转、右转、侧移
- ✅ **键盘交互**: WASD键盘控制，实时响应指令
- ✅ **GPU加速**: CUDA优化训练，290+ FPS性能
- ✅ **可视化渲染**: 实时MuJoCo 3D渲染
- ✅ **完整分析**: 训练日志记录和结果分析

## 🛠 技术栈

- **深度学习**: PyTorch 2.5.1+cu121
- **物理仿真**: MuJoCo 3.3.7
- **强化学习**: PPO (Proximal Policy Optimization)
- **数值计算**: NumPy 1.26.4
- **可视化**: OpenCV, Matplotlib
- **环境管理**: uv包管理器

## 🚀 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU (推荐，支持CUDA 11.8/12.1)
- Windows/Linux/macOS

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/xiaoheilong3112/t1_robot.git
cd t1_robot
```

2. **创建虚拟环境**
```bash
uv venv .venv
```

3. **激活环境并安装依赖**

Windows:
```bash
.venv\Scripts\activate
uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Linux/macOS:
```bash
source .venv/bin/activate
uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

4. **GPU支持 (可选但推荐)**
```bash
# CUDA 12.1版本
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8版本
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 运行演示

**Windows用户 (推荐)**:
```bash
quick_demo.bat
```

**所有平台**:
```bash
# 训练模型
python train_script.py

# 测试模型
python test_script.py

# 完整演示
python demo_script.py
```

## 📁 项目结构

```
t1_robot/
├── models/
│   └── t1_robot.xml           # T1机器人MuJoCo模型
├── checkpoints/               # 训练模型保存目录
│   └── final_model.pt        # 最终训练模型
├── logs/                     # 训练日志
│   └── training_log.json     # 详细训练记录
├── t1_robot_env.py           # T1机器人仿真环境
├── ppo_algorithm.py          # PPO算法实现
├── train_script.py           # 训练脚本
├── test_script.py            # 测试脚本
├── demo_script.py            # 演示脚本
├── analysis_tools.py         # 结果分析工具
├── pyproject.toml            # 项目配置
└── README.md                 # 项目说明
```

## 🎮 操作说明

### 键盘控制

运行`test_script.py`或`demo_script.py`后：

- **W**: 前进加速
- **S**: 后退/减速
- **A**: 左转
- **D**: 右转
- **Q**: 向左侧移
- **E**: 向右侧移
- **Space**: 立即停止所有运动
- **R**: 重置所有指令

## 📊 性能指标

### 训练结果
- **训练步数**: 50,000步 (约3分钟)
- **平均奖励**: -612.22 ± 17.88
- **稳定步数**: 11.7 ± 0.5步
- **训练FPS**: 290+ (GPU加速)

### 系统要求
- **最低配置**: CPU训练，4GB RAM
- **推荐配置**: NVIDIA GPU，8GB+ VRAM，16GB+ RAM
- **最佳性能**: RTX系列GPU + CUDA 12.1

## 🔧 开发说明

### 核心模块

1. **T1RobotEnv** (`t1_robot_env.py`)
   - MuJoCo仿真环境封装
   - 观察空间：45维 (关节位置、速度、IMU数据等)
   - 动作空间：12维 (各关节力矩控制)
   - 多维度奖励函数设计

2. **PPOAgent** (`ppo_algorithm.py`)
   - Actor-Critic网络架构
   - PPO裁剪优化算法
   - GPU加速训练支持
   - 数值稳定性优化

3. **训练流程** (`train_script.py`)
   - 经验收集和回放
   - GAE优势估计
   - 小批量更新机制
   - 自动检查点保存

### 技术亮点

- **数值稳定性**: Tanh激活函数防止梯度爆炸
- **GPU优化**: 充分利用并行计算资源
- **模块化设计**: 清晰的接口和职责分离
- **错误处理**: 完善的异常处理和日志记录

## 🐛 常见问题

### 安装问题

**Q: 安装依赖时报错**
```bash
# 尝试使用不同的镜像源
uv pip install -e . --index-url https://mirrors.aliyun.com/pypi/simple/
```

**Q: CUDA不可用**
```bash
# 检查CUDA安装
python -c "import torch; print(torch.cuda.is_available())"

# 安装CPU版本
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 运行问题

**Q: 找不到模型文件**
```bash
# 确保先进行训练
python train_script.py
```

**Q: MuJoCo渲染错误**
```bash
# 检查显示器配置
export DISPLAY=:0  # Linux
# 或尝试无头模式
python test_script.py --no_render
```

## 📈 进阶功能 (未来规划)

- [ ] **复杂地形适应**: 楼梯、斜坡、障碍物
- [ ] **抗扰动能力**: 外力干扰下的平衡恢复
- [ ] **自然语言控制**: 语音指令解析和执行
- [ ] **自主恢复**: 跌倒后自动站立
- [ ] **多任务学习**: 同时学习多种运动技能

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支: `git checkout -b feature/新功能`
3. 提交更改: `git commit -am '添加新功能'`
4. 推送分支: `git push origin feature/新功能`
5. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 👨‍💻 作者

**xiaoheilong3112**
- GitHub: [@xiaoheilong3112](https://github.com/xiaoheilong3112)

## 🙏 致谢

- [OpenAI](https://openai.com/) - PPO算法论文
- [DeepMind](https://deepmind.com/) - MuJoCo物理引擎
- [PyTorch团队](https://pytorch.org/) - 深度学习框架
- [Motphys](https://www.motphys.com/) - 技术测试机会

## 📞 联系方式

如有问题或合作意向，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至项目维护者

---

⭐ 如果这个项目对你有帮助，请给个Star支持！

**项目状态**: ✅ 核心功能完成，持续优化中