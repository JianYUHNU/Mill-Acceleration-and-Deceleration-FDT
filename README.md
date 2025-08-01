# Mill-Acceleration-and-Deceleration-FDT
Based on the existing FDT data, read the CSV file, conduct data analysis, and present the visualization results.
项目简介
本项目旨在通过机器学习方法构建 FDT（轧机）出口温度的回归预测模型。代码实现了数据预处理、特征自动选择、模型训练与评估的完整流程，并集成了 TensorBoard 可视化和模型权重自动管理功能。

This project aims to build a regression prediction model for FDT (rolling mill) outlet temperature using machine learning methods. The code implements a complete workflow including data preprocessing, automatic feature selection, model training and evaluation, with integrated TensorBoard visualization and automatic model weight management.
主要功能
自动剔除无关特征（结合手动筛选和统计规则）
数据标准化与划分（训练集 / 验证集）
神经网络模型构建与训练
TensorBoard 可视化支持
模型权重自动保存与加载
模型结构与参数统计展示
Key Features
Automatic removal of irrelevant features (combining manual screening and statistical rules)
Data standardization and splitting (training/validation sets)
Neural network model construction and training
TensorBoard visualization support
Automatic model weight saving and loading
Model structure and parameter statistics display
环境依赖
Python 3.x
pandas, numpy
PyTorch
scikit-learn
tqdm
torchinfo
tensorboard
Environment Dependencies
Python 3.x
pandas, numpy
PyTorch
scikit-learn
tqdm
torchinfo
tensorboard
项目结构
plaintext
.
├── main.py          # 主程序入口，包含完整流程
├── model.py         # 模型定义
├── loss.py          # 损失函数定义
├── train.py         # 训练逻辑实现
├── best_model.pt    # 训练好的最佳模型权重（自动生成）
└── selected_features.csv  # 筛选后的特征列表（自动生成）
Project Structure
plaintext
.
├── main.py          # Main program entry with complete workflow
├── model.py         # Model definition
├── loss.py          # Loss function definition
├── train.py         # Training logic implementation
├── best_model.pt    # Trained best model weights (auto-generated)
└── selected_features.csv  # Selected feature list (auto-generated)
使用方法
配置数据路径（ROOT_DIR）指向包含 CSV 数据的文件夹
运行主程序：python main.py
查看可视化结果：tensorboard --logdir runs
Usage
Configure the data path (ROOT_DIR) to point to the folder containing CSV data
Run the main program: python main.py
View visualization results: tensorboard --logdir runs
模型说明
模型采用简单的前馈神经网络结构：

输入层：根据筛选后的特征数量动态确定
隐藏层：128 维 -> 64 维（均使用 ReLU 激活函数）
输出层：1 维（预测温度值）
损失函数：L1 损失（MAE）
优化器：Adam
Model Description
The model uses a simple feedforward neural network structure:

Input layer: dynamically determined based on the number of selected features
Hidden layers: 128D -> 64D (both using ReLU activation function)
Output layer: 1D (predicted temperature value)
Loss function: L1 loss (MAE)
Optimizer: Adam
特征选择流程
手动剔除明显无关的列（日期、时间等）
剔除缺失率超过 1% 的特征
基于方差阈值过滤低方差特征
保留与目标变量相关性超过阈值的特征
Feature Selection Process
Manually remove obviously irrelevant columns (date, time, etc.)
Remove features with missing rate exceeding 1%
Filter low-variance features based on variance threshold
Keep features with correlation to target variable exceeding threshold
