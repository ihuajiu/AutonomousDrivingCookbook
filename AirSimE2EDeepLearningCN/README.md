# 使用端到端深度学习的自主驾驶：AirSim教程

### 作者：
**[Mitchell Spryn](https://www.linkedin.com/in/mitchell-spryn-57834545/)**, Software Engineer II, Microsoft

**[Aditya Sharma](https://www.linkedin.com/in/adityasharmacmu/)**, Program Manager, Microsoft


## Overview

在本教程中，您将学习如何使用从AirSim仿真环境。您将培训一个模型，以学习如何驾驶汽车通过部分山/景观地图在[AirSim simulation environment](https://github.com/Microsoft/AirSim)使用一个单一的正面摄像头视觉输入。这样的任务通常被认为是自动驾驶的“你好世界”，但是在完成本教程之后，你将有足够的背景来开始探索你自己的新想法。通过本教程的篇幅，您还将学习到与端到端深度学习方法合作的一些实际方面和细微之处。

下面是该模型的一个简短示例：

![car-driving](car_driving.gif)

## 本教程的结构

本教程中提供的代码是在[Keras](https://keras.io/)，一个高级的深入学习Python api，能够运行在[CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/), [TensorFlow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/index.html)。Keras允许您使用您选择的深度学习框架，以及它的简单使用，这使它成为初学者的理想选择，消除了最流行框架附带的学习曲线。

本教程以Python记事本的形式提供给您。Python笔记本使您可以轻松地阅读说明和解释，并在同一个文件中编写和运行代码，所有这些都可以轻松地在浏览器窗口中工作。您将按顺序阅读下列记事本：

**[数据挖掘与准备](DataExplorationAndPreparation.ipynb)**

**[训练模型](TrainModel.ipynb)**

**[测试模型](TestModel.ipynb)**

如果您以前从未使用过Python notebooks，我们强烈建议您[查看文档](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).


## 先决条件和设置
### 所需背景

你应该熟悉神经网络和深度学习的基础知识。你不需要知道像LSTM或强化学习这样的高级概念，但是你应该知道卷积神经网络是如何工作的。要在短时间内获得一个强大的背景，一个非常好的起点是这本强烈推荐的关于[这个主题的书](http://neuralnetworksanddeeplearning.com/) 作者：Michael Nielsen。它是免费的，非常短，可以在网上获得。它可以在不到一周的时间内为你提供一个坚实的基础。

您还应该熟悉Python。至少，您应该能够阅读和理解用Python编写的代码。

### 环境设置
1. 安装[Anaconda](https://conda.io/docs/user-guide/install/index.html)使用Python3.5或更高版本。
2. 安装[CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine)或安装[Tensorflow](https://www.tensorflow.org/install/install_windows)
3. 安装[h5py](http://docs.h5py.org/en/latest/build.html)
4. 安装[Keras](https://keras.io/#installation)和[配置Keras后端](https://keras.io/backend/)使用TensorFlow(默认)或CNTK。
5. 安装[AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy)。确保将AzCopy可执行文件的位置添加到系统路径。
6. 安装其他依赖项。在anaconda环境中，以root或管理员身份运行[InstallPackages.py](AirSimE2EDeepLearning\InstallPackages.py)。这会将下列软件包安装到您的环境中：
    * jupyter
    * matplotlib v. 2.1.2
    * image
    * keras_tqdm
    * opencv
    * msgpack-rpc-python
    * pandas
    * numpy
    * scipy

**安装环境附加说明**
- 因为项目比较久远，当时使用的依赖包对于现在来说都比较古老，会出现很多错误，因此需要指定依赖包的安装版本来构建环境
    - python==3.6
    * tensorflow==1.5.0
    * keras==2.1.2
    * h5py==2.10.0
    * tornado==4.5

### 模拟器包
我们已经为本食谱中的教程创建了AirSim模拟环境的独立构建。您可以[从这里下载构建包](https://airsimtutorialdataset.blob.core.windows.net/e2edl/AD_Cookbook_AirSim.7z)。考虑使用[AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy)，因为文件大小很大。下载包后，解压缩它并运行PowerShell命令

`
.\AD_Cookbook_Start_AirSim.ps1 landscape
`

在landscape环境中启动模拟器


### 硬件
强烈建议使用GPU来运行本教程中的代码。虽然只需要一个CPU就可以训练模型，但是完成训练需要很长时间。本教程是用NVIDIA GTX970 GPU开发的，训练时间为45分钟。

如果您没有可用的GPU，则可以将[基于Azure的深度学习VM(已不可用)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning)，它附带了安装的所有依赖项和库(如果使用此VM，请使用所提供的py35环境)


### 数据集

模型的数据集相当大。你可以[从这里下载](https://aka.ms/AirSimTutorialDataset)。第一个笔记本将提供关于如何访问数据的指导，一旦你下载了它。最后的未压缩数据集大小约为3.25GB(虽然与训练实际的自动驾驶汽车所需的千兆字节数据相比，这还不够，但对于本教程来说应该足够了)。


### 作者的便条

我们已经尽了最大的努力确保本教程能够帮助您开始学习自主驾驶的基础知识，并使您能够开始独立探索新的想法。我们希望听到您对我们如何改进和发展本教程的反馈意见。我们还想知道我们还可以为您提供哪些其他的教程，帮助您推进您的职业目标。请使用GitHub问题部分获得所有反馈。所有反馈都将受到密切监测。如果你有想法的话[合作](../README.md#contributing)继续，请随时与我们联系，我们将很高兴与您合作。

