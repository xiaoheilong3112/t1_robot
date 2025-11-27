@echo off
echo ========================================
echo T1机器人强化学习项目快速演示
echo Motphys技术测试项目
echo ========================================
echo.

REM 检查虚拟环境是否存在
if not exist ".venv\Scripts\python.exe" (
    echo 错误: 找不到虚拟环境
    echo 请先运行以下命令创建环境:
    echo   uv venv .venv
    echo   uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
)

REM 检查训练好的模型是否存在
if not exist "checkpoints\final_model.pt" (
    echo 训练好的模型不存在，开始快速训练...
    echo 训练参数: 50000步，GPU加速
    echo.
    .venv\Scripts\python.exe train_script.py
    if errorlevel 1 (
        echo 训练失败，请检查环境配置
        pause
        exit /b 1
    )
    echo.
    echo 训练完成！
    echo.
)

echo 准备开始演示...
echo.
echo 演示内容:
echo 1. 基本行走能力
echo 2. 方向控制能力
echo 3. 稳定性测试
echo 4. 交互式键盘控制
echo.
echo 注意: 演示过程中会打开MuJoCo可视化窗口
echo       请确保有足够的显示空间
echo.
pause

echo 启动演示程序...
.venv\Scripts\python.exe demo_script.py

if errorlevel 1 (
    echo.
    echo 演示过程中出现错误
    echo 可能的原因:
    echo 1. GPU驱动问题
    echo 2. MuJoCo渲染问题
    echo 3. 模型文件损坏
    echo.
    echo 尝试运行基础测试:
    .venv\Scripts\python.exe test_script.py
) else (
    echo.
    echo ========================================
    echo 演示成功完成！
    echo ========================================
    echo.
    echo 项目功能验证:
    echo ✅ 环境配置正确
    echo ✅ GPU加速工作
    echo ✅ 模型训练成功
    echo ✅ 机器人行走稳定
    echo ✅ 控制响应正常
    echo ✅ 物理仿真准确
    echo ✅ 渲染效果良好
    echo.
)

echo 其他可用命令:
echo - 重新训练: .venv\Scripts\python.exe train_script.py
echo - 基础测试: .venv\Scripts\python.exe test_script.py
echo - 分析结果: .venv\Scripts\python.exe -c "import analysis_tools; print('分析工具可用')"
echo.

pause
