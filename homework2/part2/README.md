# Usage
### LeNet
```shell
# Train LeNet model
$ python3 main.py --model LeNet    

# Evaluate LeNet on public test set
$ python3 eval.py --model LeNet --path ./save_dir/LeNet/best_model.pt
```

### myResnet
```shell
# Train myResnet model
$ python3 main.py --model myResnet                 

# Evaluate myResnet on public test set
$ python3 eval.py --model myResnet --path ./save_dir/myResnet/best_model.pt              
```