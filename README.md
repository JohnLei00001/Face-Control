# Face-Control (Head & Eye Mouse Controller)

**⚠️ 项目状态：开发中 (Work in Progress)**

本项目是一个基于 Python 的头部和眼部动作追踪鼠标控制器。它通过摄像头捕捉用户的面部动作，实现用头部移动控制鼠标指针，以及通过眨眼或悬停（Dwell）来触发鼠标点击。

## 主要特性 (Features)

- **头部追踪控制鼠标**：通过面部移动来控制屏幕上的鼠标指针。
- **眨眼点击 (Blink to Click)**：支持通过眨眼动作来触发鼠标左键点击或双击。
- **悬停点击 (Dwell to Click)**：当鼠标在某个区域停留一定时间后自动触发点击。
- **平滑过滤**：内置 OneEuro 滤波器，提供平滑、稳定的鼠标控制体验。
- **高可定制性**：支持通过命令行参数调整灵敏度、平滑度、死区、眨眼检测阈值等。
- **双引擎支持**：支持 OpenCV 和 MediaPipe 进行面部及关键点检测。

## 依赖环境 (Requirements)

请确保你的环境已安装 Python 3.x。安装项目所需的依赖：

```bash
pip install -r requirements.txt
```

主要的依赖包包括：
- `opencv-contrib-python`
- `numpy`
- `pynput`
- `mediapipe`

## 使用说明 (Usage)

直接运行主脚本启动程序。默认情况下，摄像头画面窗口是隐藏的，程序会在后台运行：

```bash
python src/head_mouse.py
```

### 常用参数

如果你想显示摄像头画面并查看追踪效果，可以添加 `--show` 参数：

```bash
python src/head_mouse.py --show
```

更多自定义参数（例如调整灵敏度、开启/关闭悬停点击等）：

- `--sensitivity 3.5`：设置鼠标移动灵敏度（默认 3.5）
- `--smoothing 0.6`：设置平滑度（默认 0.6）
- `--dwell` 或 `--no-dwell`：开启或关闭悬停点击功能
- `--blink_click` 或 `--no-blink_click`：开启或关闭眨眼点击功能
- `--camera 0`：指定使用的摄像头索引（默认 0）

要查看所有支持的命令行参数，请运行：

```bash
python src/head_mouse.py --help
```

### 退出程序

在运行程序的终端中按下 `Ctrl + C` 即可退出程序。

## 目录结构 (Structure)

- `src/head_mouse.py` - 核心控制脚本
- `requirements.txt` - Python 依赖列表
- `.gitignore` - Git 忽略文件配置

## 贡献 (Contributing)

本项目仍在积极开发中。欢迎提交 Issue 或 Pull Request 来帮助改进！
