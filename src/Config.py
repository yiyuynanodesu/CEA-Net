import os

class config:
    # hyperparam
    epoch = 10
    learning_rate = 3e-5
    weight_decay = 0
    num_labels = 4
    loss_weight = [1.68, 9.3, 3.36]

    # Fuse
    fuse_model_type = 'NaiveCombine'
    only = None
    middle_hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128

    # BERT
    fixed_text_model_params = False
    bert_name = 'roberta-base'
    bert_learning_rate = 5e-6
    bert_dropout = 0.2

    # ResNet
    resnet_name = 'resnet50d'
    fixed_img_model_params = False
    image_size = 224
    fixed_image_model_params = True
    resnet_learning_rate = 5e-6
    resnet_dropout = 0.2
    img_hidden_seq = 64


    # Dataloader params
    checkout_params = {'batch_size': 4, 'shuffle': False}
    train_params = {'batch_size': 4, 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': 4, 'shuffle': False, 'num_workers': 0}
    test_params =  {'batch_size': 4, 'shuffle': False, 'num_workers': 0}

    
    