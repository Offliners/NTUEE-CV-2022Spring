# Homework 2 - Scene Recognition & Image Classification Using CNN
Detail : [Link](hw2.pdf)

## Usage
```shell
# Build anaconda virtual environment
conda create --name dlcv_hw2 python=3.7.6
conda activate dlcv_hw2

# Install cyvlfeat
conda install -c conda-forge cyvlfeat

# Install required package
pip3 install -r requirements.txt
```

## Part 1
### Plot confusion matrix of two settings. (i.e. Bag of sift and tiny image representation)
|Feature|Confusion Matrix|
|-|-|
|Tiny Image (Accuracy : `24.2%`)|![Tiny Image](./part1/tiny_image.png)|
|Bag of SIFT (Accuracy : `60.7%`)|![Bag of SIFT](./part1/bag_of_sift.png)|

## Part 2
### Compare the performance on residual networks and LeNet. Plot the learning curve (loss and accuracy) on both training and validation sets for both 2 schemes. 8 plots in total

### LeNet
||Loss|Accuracy|
|-|-|-|
|Training|![loss.png](./part2/save_dir/LeNet/train/loss.png)|![acc.png](./part2/save_dir/LeNet/train/acc.png)|
|Validation|![loss.png](./part2/save_dir/LeNet/valid/loss.png)|![acc.png](./part2/save_dir/LeNet/valid/acc.png)


My Resnet (without pseudo label)
||Loss|Accuracy|
|-|-|-|
|Training|![loss.png](./part2/save_dir/myResnet/train/loss.png)|![acc.png](./part2/save_dir/myResnet/train/acc.png)|
|Validation|![loss.png](./part2/save_dir/myResnet/valid/loss.png)|![acc.png](./part2/save_dir/myResnet/valid/acc.png)