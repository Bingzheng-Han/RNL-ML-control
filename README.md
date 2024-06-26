# RNL-ML-control

The practical implementation of machine learning in flow control is limited due to its significant training expenses. In the present study, the convolutional neural network (CNN) trained with the data of restricted nonlinear model (RNL) is used to predict the normal velocity on a detection plane at $y^+=10$ in turbulent channel flow, and the predicted velocity is used as wall blowing and suction for drag reduction.
Active control test is carried out by using the well-trained CNN in direct numerical simulation (DNS). Substantial drag reduction rates up to 19\% and 16\% are obtained based on the spanwise and streamwise wall shear stresses, respectively. 
Furthermore, we explore the online control of wall turbulence by combining RNL with reinforcement learning (RL).
The RL is constructed to determine the optimal wall blowing and suction based on its observation of the wall shear stresses without using the label data on the detection plane for training. The controlling and training processes are conducted synchronously in RNL flow field.
Control strategy discovered by RL has similar drag reduction rates  with those obtained previously by the established method.
Also, the training cost decreases by over thirty times at $Re_{\tau}=950$ compared to the DNS-RL model.
The present results provide a perspective that combining RNL with machine learning control for drag reduction in wall turbulence can be effective and computationally economical. Also, this approach can be easily extended to flows at higher Reynolds numbers.

Here, codes of the CNN and RL methods are presented. An interface function of machine learning codes is also given so that any fluid solvers can easily call the functions of them.
