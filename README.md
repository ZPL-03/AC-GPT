# 航空复合材料多尺度智能设计平台

## 项目简介

本软件是一个集成了深度学习与多尺度建模技术的航空复合材料智能设计平台,面向航空航天领域复合材料结构的全流程设计与优化。平台采用PyQt6框架开发,整合了TensorFlow深度学习引擎,实现了从微结构到宏观结构的多尺度协同设计。

## 主要功能

### 1. 多尺度设计层级

- **微结构层级**: 基于CNN和ICNN模型的微观结构分析与性能预测
- **层合板层级**: 复合材料层合板铺层设计、刚度矩阵计算与强度分析
- **加筋结构层级**: 加筋板/壳结构几何建模、屈曲分析与优化设计
- **机身段层级**: 飞机机身段级结构建模、载荷分析与结构评估

### 2. AI驱动的智能设计

- **CNN图像特征提取**: 自动识别微结构图像的纤维分布特征
- **物理信息神经网络**: 基于物理约束的材料性能预测模型
- **迁移学习**: 支持在预训练模型基础上进行微调,适应任意特定材料体系
- **模型优化**: 内置多种优化算法,实现参数自动寻优

### 3. 材料数据库

- 内置常用航空复合材料数据库(碳纤维、玻璃纤维、芳纶纤维等)
- 支持自定义材料属性输入
- 材料性能可视化展示

### 4. 可视化与分析

- 2D/3D结构可视化
- 应力应变场云图显示
- 失效包络线与载荷路径图
- 实时参数敏感性分析

### 5. 项目管理

- 支持项目保存与加载(.json格式)
- 设计参数导入导出(Excel/CSV)
- 计算结果批量导出
- 完整的操作日志记录

## 技术架构

- **界面框架**: PyQt6
- **深度学习**: TensorFlow 2.x + Keras
- **科学计算**: NumPy, SciPy, Pandas
- **图像处理**: OpenCV
- **数据分析**: Scikit-learn
- **可视化**: Matplotlib, Plotly

## 系统要求

### 最低配置
- 操作系统: Windows 10/11, Linux, macOS
- CPU: Intel Core i5或同等性能处理器
- 内存: 8GB RAM
- 硬盘: 2GB可用空间
- Python: 3.8+

### 推荐配置
- 操作系统: Windows 10/11 (64位)
- CPU: Intel Core i7或更高
- 内存: 16GB RAM及以上
- GPU: NVIDIA GPU (支持CUDA 11.0+,用于深度学习加速)
- 硬盘: 5GB可用空间
- Python: 3.9-3.11

## 安装指南

### 1. 克隆仓库

```bash
git clone https://github.com/ZPL-03/composite-design-platform.git
cd composite-design-platform
```

### 2. 创建虚拟环境(推荐)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备资源文件

确保在项目根目录下创建以下文件夹结构:
```
project_root/
├── assets/
│   ├── logo.ico
│   └── welcome.png
├── models/
│   ├── pretrained/
│   └── finetuned/
└── data/
    ├── materials/
    └── examples/
```

## 快速开始

### 启动程序

```bash
python GUI-V1_0.py
```

### 基本工作流程

1. **创建新项目**: 点击"新建项目"按钮,设置项目名称和设计层级
2. **选择材料**: 从材料数据库中选择纤维和基体材料
3. **配置模型**: 选择预训练模型或进行模型微调
4. **设计分析**: 
   - 微结构: 上传微观图像,进行分割与性能预测
   - 层合板: 定义铺层序列,计算刚度矩阵
   - 加筋结构: 建立几何模型,进行屈曲分析
   - 机身段: 定义载荷边界条件,评估结构响应
5. **结果查看**: 在可视化面板查看分析结果
6. **导出数据**: 导出计算结果和可视化图表

## 深度学习模型

### 预训练模型

平台提供以下预训练模型:

- **CNN_Segmentation_v1.h5**: 微结构图像分割模型
- **ICNN_Prediction_v1.h5**: 材料性能预测模型
- **Laminate_Analysis_v1.h5**: 层合板分析模型

### 模型微调

支持用户使用自己的数据集对预训练模型进行微调:

1. 准备训练图像(支持jpg, png格式)
2. 准备对应的参数表格(Excel或CSV格式)
3. 在"模型微调"面板配置训练参数
4. 点击"开始训练"进行微调
5. 训练完成后模型自动保存到finetuned目录

## 项目文件格式

### 项目文件(.json)

项目文件包含完整的设计配置和状态信息:

```json
{
  "app": "MultiscaleDesignPlatform",
  "version": 1,
  "project": {
    "name": "项目名称",
    "level": "微结构",
    "units": "SI",
    "model_mode": "预训练模型",
    "pretrained_models": {},
    "finetuned_models": {}
  },
  "pages": {
    "microstructure": {},
    "laminate": {},
    "stiffened": {},
    "fuselage": {}
  }
}
```

### 输入数据格式

- **微结构图像**: JPG, PNG (推荐分辨率: 512×512或1024×1024)
- **材料参数**: CSV, XLSX格式,需包含必要的力学性能参数
- **铺层序列**: 角度数组,例如 [0, 45, -45, 90]

### 输出数据格式

- **分析报告**: TXT格式
- **可视化图表**: PNG, JPG格式
- **数据表格**: CSV, XLSX格式

## 常见问题

### Q1: 软件无法启动

- 检查Python版本是否为3.8+
- 确认所有依赖包已正确安装
- 查看终端/命令行输出的错误信息

### Q2: GPU未被识别

- 安装NVIDIA CUDA Toolkit (11.0+)
- 安装cuDNN库
- 安装GPU版本的TensorFlow: `pip install tensorflow-gpu`

### Q3: 模型加载失败

- 确认models目录下存在对应的.h5模型文件
- 检查模型文件是否损坏
- 尝试重新下载预训练模型

### Q4: 图像分割结果不准确

- 确保输入图像质量良好,对比度适中
- 考虑使用您自己的数据集进行模型微调
- 调整图像预处理参数

### Q5: 内存不足错误

- 减小批处理大小
- 降低图像分辨率
- 关闭其他占用内存的程序

## 开发路线图

- [ ] 增加更多材料类型的预训练模型
- [ ] 集成更先进的优化算法(遗传算法、粒子群优化等)
- [ ] 支持有限元分析接口(ABAQUS, ANSYS等)
- [ ] 添加云端模型训练功能
- [ ] 开发移动端应用(iOS/Android)
- [ ] 支持多语言界面(英文、中文)

## 贡献指南

欢迎对本项目做出贡献!请遵循以下步骤:

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

### 代码规范

- 遵循PEP 8 Python编码规范
- 添加适当的注释和文档字符串
- 编写单元测试以确保代码质量
- 提交前进行代码审查

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 引用

如果您在研究或项目中使用了本软件,请引用:

```
@software{composite_design_platform_2025,
  title={航空复合材料多尺度智能设计平台},
  author={Zhengpeng Liu},
  year={2025},
  url={https://github.com/yourusername/composite-design-platform}
}
```

## 联系方式

- 项目主页: https://github.com/yourusername/composite-design-platform
- 问题反馈: https://github.com/yourusername/composite-design-platform/issues
- 电子邮件: your.email@example.com

## 致谢

感谢以下开源项目和社区的支持:

- TensorFlow团队提供的深度学习框架
- PyQt6团队提供的GUI框架
- 科学计算社区提供的NumPy、SciPy等工具
- 所有为本项目做出贡献的开发者和用户

## 更新日志

### Version 1.0 (当前版本)
- 完整的多尺度设计功能
- 集成CNN和ICNN深度学习模型
- 模型微调功能
- 项目管理系统
- 可视化增强
- 性能优化

---

**注意**: 本软件为研究和教育目的开发,用于实际工程应用时请进行充分的验证和测试。
