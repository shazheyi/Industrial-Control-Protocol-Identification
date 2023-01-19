# Industrial-Control-Protocol-Identification

一、文件说明

（1）ICP.h5 为由数据预处理生成的工控协议数据集文件，包括工控协议生成图像和相应的标签。

（2）disICPcnn.py 文件为分布式卷积神经网络工控协议识别模型训练文件。

二、环境说明

（1）模拟边缘设备：四个树莓派4B单板机；

（2）操作系统：基于Debian的Raspberry Pi OS

（3）Tensorflow：2.3.0版本

三、运行说明

（1）打开需要使用的树莓派设备，使用网线直连，并查找每个树莓派的IP地址，在disICPcnn.py中更改相应的ps_hosts和worker_hosts。

（2）先启动ps树莓派，在对应的ps树莓派上运行disICPcnn.py文件，运行命令为

#python3 disICPcnn.py --job_name=ps --task_id=0

（3）再启动worker树莓派，在对应的worker树莓派上运行disICPcnn.py文件，其中task_id为相应的worker编号，运行命令为

#python3 disICPcnn.py --job_name=worker --task_id=0
