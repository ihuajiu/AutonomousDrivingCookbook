# AutonomousDrivingCookbook开箱即用手册

## 1. 介绍
基于微软AirSim模拟器的自动驾驶入门项目汉化版

原项目地址：[https://github.com/Microsoft/AutonomousDrivingCookbook](https://github.com/Microsoft/AutonomousDrivingCookbook)

项目面向自动驾驶初学者、研究人员和行业专家。项目以jupter notebook作为载体，使用流行的开源工具(如Keras、TensorFlow等)构建，项目提供数据集、源代码、AirSim模拟器，以便于实现自动驾驶快速仿真。

------
<p align="center">
  <img src="AirSimE2EDeepLearning/car_driving.gif?raw=true" />
</p>

## 2. 软件架构
目前，有以下教程：

- [使用端到端深度学习的自主驾驶：AirSim教程](./AirSimE2EDeepLearning)
- [分布式深度强化学习在自主驾驶中的应用](./DistributedRL)
- 以下教程不久将面世：
    - 基于深度学习的车道检测


## 3. 安装教程
### 3.1 安装运行环境

- 切换conda国内安装源
    - Anaconda官方软件包更新速度太慢，建议配置国内软件源
      - 在Win操作系统用户目录下有一个.condarc文件，替换文件内所有内容
      ```
      channels:
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
      ssl_verify: true
      ```
  - 还原默认官方安装源
      ```
      conda config --remove-key channels
      ```
- 切换pip安装源  
  - 永久配置pip安装源
      - 做如下配置后，无需再使用-i选项
          ```
          pip install pip -U #升级 pip 到最新的版本后进行配置：
          pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
          ```

  - 还原默认pip安装源
      ```
      pip config unset global.index-url
      ```
- 安装依赖包
    - 创建conda虚拟环境

    为了减少其它库的冲突问题，强烈建立新建一个虚拟环境
    ```
    # python版本务必选择3.6，否则运行过程中的冲突会让你怀疑人生
    conda create -n airsim python=3.6
    # 切换aisim虚拟环境
    conda activate airsim
    ```
    - 使用InstallPackages.py安装依赖脚本
    ```
    # 推荐使用命令运行脚本，IDE工具有时候选择的虚拟环境不对
    python InstallPackages.py
    ```

    **注意事项：**
    因为项目过于久远，安装现在的依赖包会和代码冲突，因此需要指定版本安装解决这个问题,InstallPackages.py已经修改为对应的版本。
   
    - tensorflow==1.5.0 
      - 安装tensorflow1.x版本，预防keras版本冲突问题
      - 需要安装特定版本1.5.0，否则报错:tensorflow_backend.py:64: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_d
    - keras==2.1.2
      - 必须使用2.1.2版本，否则会报错:ValueError: `brightness_range should be tuple or list of two floats. Received: 0.0, https://github.com/microsoft/AutonomousDrivingCookbook/issues/89
    - tornado==4.5
      - 解决ZMQILoop和float变量*运算的问题
    - h5py==2.10.0
      - 解决警告问题
    - 如果vscode启动内核失败，则执行如下命令
  ```
    conda install -n airsim ipykernel --update-deps --force-reinstall
  ```
至次环境搭建完毕。

## 4. 使用说明

### 4.1 下载模拟器

我们已经为本食谱中的教程创建了AirSim模拟环境的独立构建。您可以[从这里下载构建包](https://airsimtutorialdataset.blob.core.windows.net/e2edl/AD_Cookbook_AirSim.7z)。考虑使用[AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy)，因为文件大小很大。

### 4.2 下载数据集

模型的数据集相当大。你可以[从这里下载](https://aka.ms/AirSimTutorialDataset)。第一个笔记本将提供关于如何访问数据的指导，一旦你下载了它。最后的未压缩数据集大小约为3.25GB(虽然与训练实际的自动驾驶汽车所需的千兆字节数据相比，这还不够，但对于本教程来说应该足够了)。

### 4.3 教程使用步骤
#### 4.3.1 AirSimE2EDeepLearning端到端的深度学习
##### 4.3.1.1 配置原始数据集目录和预处理后的数据集目录
打开[DataExplorationAndPreparation.ipynb](./AirSimE2EDeepLearning/DataExplorationAndPreparation.ipynb)文件修改你的电脑上的实际目录
```
# << 配置下载的数据集目录 >>
RAW_DATA_DIR = 'data_raw/'

# << 配置预处理后(*.h5)的输出目录 >>
COOKED_DATA_DIR = 'data_cooked/'
```

##### 4.3.1.2 配置Step1中的[配置预处理后(*.h5)的输出目录]和模型文件保存目录
打开[TrainModel.ipynb](./AirSimE2EDeepLearning/TrainModel.ipynb)修改如下目录

```
# << 配置前一步预处理好的数据集目录 >>
COOKED_DATA_DIR = 'data_cooked/'

# << 模型文件输出目录：随着梯度越来越小，模型会逐步更新 >>
MODEL_OUTPUT_DIR = 'model' # 不建议修改，保持默认即可
```

##### 4.3.1.3 模型预处理、训练、推理预测及模拟器自动驾驶运行测试
- 数据挖掘与准备
    依次执行[DataExplorationAndPreparation.ipynb](./AirSimE2EDeepLearning/DataExplorationAndPreparation.ipynb)，完成数据预处理
  
- 模型训练
    依次执行[TrainModel.ipynb](./AirSimE2EDeepLearning/TrainModel.ipynb)，完成模型训练

- 启动AirSim模拟器
  - 启动Powershell
    第一次启动，用管理员权限打开Powershell shell，首先配置权限，否则会提示：**禁止运行脚本**
    ```
    set-ExecutionPolicy RemoteSigned
    ```
    以后启动powershell就不用管理员权限了。
  - 启动AirSim模拟器
    在Powershell中定位到模拟器安装位置AD_Cookbook_AirSim，执行如下命令启动
    ```
    .\AD_Cookbook_Start_AirSim.ps1 landscape
    ```
    landscape是数据集对应的场景，还有City、Hawii、Neighborhood三种场景。

- 启动模型推理及自动驾驶
  - 执行[TestModel.ipynb](./AirSimE2EDeepLearning/TestModel.ipynb)启动自动驾驶模拟
    - 注意事项：TestModel.ipynb有可能在连上AisSim模拟器的时候卡死，应该是vscode jupyter工具的问题，将代码摘出来，直接运行python文件即可运行[Python版TestModel](./AirSimE2EDeepLearning/TestModel.py)。 
  
### 4.2 分布式强化学习
待添加
