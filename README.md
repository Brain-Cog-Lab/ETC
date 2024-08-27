# Improving Stability and Performance of Spiking Neural Networks through Enhancing Temporal Consistency
This repository contains code form our paper Improving Stability and Performance of Spiking Neural Networks through Enhancing Temporal Consistency. If you use our code or refer to this project, please cite 
```
@article{zhao2023improving,
  title={Improving stability and performance of spiking neural networks through enhancing temporal consistency},
  author={Zhao, Dongcheng and Shen, Guobin and Dong, Yiting and Li, Yang and Zeng, Yi},
  journal={arXiv preprint arXiv:2305.14174},
  year={2023}
}
```

Install the corresponding package
```shell
pip install -e .
```
You can run the corresponding code to get the result in the paper. 
```shell
python main.py --batch-size 128  --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --act-fun QGateGrad --num-classes 10 --thresh 0.5  --output-with-step --distil-loss 1.  --device 6
python main.py --batch-size 128  --model sew_resnet18 --node-type LIFNode --dataset cifar10 --step 4 --act-fun QGateGrad --num-classes 10 --thresh 0.5 --output-with-step --distil-loss 1. --device 6
```