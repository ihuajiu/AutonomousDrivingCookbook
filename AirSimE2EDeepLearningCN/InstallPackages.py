import os

# Run this script from within an anaconda virtual environment to install the required packages
# Be sure to run this script as root or as administrator.
# 注销pip的升级，会导致升级卡死。
# os.system('python -m pip install --upgrade pip')

#os.system('conda update -n base conda')
os.system('conda install jupyter')
os.system('pip install pandas')
os.system('pip install numpy')
os.system('pip install tensorflow==1.5.0')
os.system('pip install keras==2.1.2')
os.system('pip install h5py==2.10.0')
os.system('pip install tornado==4.5')
os.system('pip install matplotlib==2.1.2')
os.system('pip install image')
os.system('pip install keras_tqdm')
os.system('conda install -c conda-forge opencv')
os.system('pip install msgpack-rpc-python')

os.system('conda install scipy')
# 如果vscode的jupternotebook启动内核失败，运行该命令
# os.system('conda install -n airsim ipykernel --update-deps --force-reinstall')
