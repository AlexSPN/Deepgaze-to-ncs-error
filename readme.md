This is a minimalistic reproduction of ncs porting error

Environment:

* NCSDK V1.12.00 2018-01-10
* Python 2.7.12 - used for my scripts 
* Python 3.5.2 - seems to be used when I call mvNCCheck or mvNCCompile
* tensorflow (1.4.0)


network origin: https://github.com/mpatacchiola/deepgaze/blob/master/deepgaze/head_pose_estimation.py

use export_model_tanh.py to generate checkpoint and tensorboard_logs

my_exports - those files was created on my PC with my environment, you may compare them with our outputs.
tanh_model - contains tensorflow checkpoint, compilen ncs graph, and mvProfile output.
tanh_tensorboard_logs - visualize graph in tensorboard
