

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'LeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 128,
    'lr':0.01,
    'num_out': 10,
    'num_epoch': 200,
}

myResnet_cfg = {
    'model_type': 'myResnet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 128,
    'lr':0.01,
    'num_out': 10,
    'num_epoch': 200,
}

DLA_cfg = {
    'model_type': 'DLA',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 128,
    'lr':0.01,
    'num_out': 10,
    'num_epoch': 200,
}