Install the corresponding package
```shell
pip install -e .
```
You can run the corresponding code to get the result in the paper. 
```shell
python main.py --batch-size 128  --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --act-fun QGateGrad --num-classes 10 --thresh 0.5  --output-with-step --distil-loss 1.  --device 6
python main.py --batch-size 128  --model sew_resnet18 --node-type LIFNode --dataset cifar10 --step 4 --act-fun QGateGrad --num-classes 10 --thresh 0.5 --output-with-step --distil-loss 1. --device 6
```