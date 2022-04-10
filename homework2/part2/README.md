# Usage
### LeNet
```shell
# 跑助教的LeNet模型
$ python3 main.py --model LeNet    

# 測試LeNet在public test set的accuracy
$ python3 eval.py --model LeNet --path ./save_dir/LeNet/best_model.pt --test_anno ./p2_data/annotations/public_test_annos.json
```

### myResnet
```shell
# 跑學生的myResnet模型，已經設好預設值
$ python3 main.py                  

# 測試myResnet在public test set的accuracy
$ python3 eval.py                  
```