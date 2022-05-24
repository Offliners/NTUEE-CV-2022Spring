python3 main.py --model LeNet
python3 main.py --model myResnet

python3 eval.py --model LeNet --path ./save_dir/LeNet/best_model.pt
python3 eval.py --model myResnet --path ./save_dir/myResnet/best_model.pt