

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'LeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 32,
    'lr':0.001,
    'milestones': [20, 40],
    'num_out': 10,
    'num_epoch': 60,
    'optimizer': 'SGD',
}

myResnet_cfg = {
    'model_type': 'myResnet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 32,
    'lr':0.001,
    'milestones': [20, 40],
    'num_out': 10,
    'num_epoch': 60,
    'optimizer': 'Adam',
}

preTrained_cfg = {
    'model_type': 'preTrained',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 32,
    'lr':0.001,
    'milestones': [20, 40],
    'num_out': 10,
    'num_epoch': 60,
    'optimizer': 'Adam',
}