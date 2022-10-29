# GANomaly-tf2


## Train , Validate, Infer

```bash
# cifar10
python train.py --log_dir=log --anomaly=1 --dataset=cifar10 --isize=32 --nc=3
# mnist
python train.py --log_dir=log --anomaly=2 --dataset=mnist --isize=32 --nc=1
```

### Notebooks

- [mnist_example.ipynb](mnist_example.ipynb)

## Reference

- https://github.com/samet-akcay/ganomaly
