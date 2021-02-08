'''
Config file, read by main.py to generate data, build and train model.
'''

class Config:
    data_type = "syn1"
    dim = 11
    train_no = 10000
    test_no = 10000
    selector_hdim = 100
    predictor_hdim = 200
    lr = 1e-3
    weight_decay = 1e-4
    l2 = 1e-3
    temp_anneal = 1e-3
    iterations = 2000
    batch_size = 100
    patience = 200
    save_model = True