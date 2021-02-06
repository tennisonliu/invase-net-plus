from data_generation import generate_dataset
from model import invase_model, ce_loss_with_reg
from trainer import train_iteration
from evaluator import feature_importance_score, predict
from utils import feature_performance_metric, prediction_performance_metric
import torch
import numpy as np

class Config:
    train_no = 10000
    test_no = 10000
    dim = 11
    data_type = "syn1"
    batch_size = 1000
    selector_hdim = 200
    predictor_hdim = 200
    weight_decay = 1e-3
    lr = 1e-4
    l2 = 1e-3
    temp_anneal = 1e-3
    iterations = 10000

def main():

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    print(f"Initialising training on : {device}")

    x_train, y_train, g_train = generate_dataset(n=Config.train_no, dim=Config.dim, data_type=Config.data_type, seed=0)
    x_test, y_test, g_test = generate_dataset(n=Config.test_no, dim=Config.dim, data_type=Config.data_type, seed=0)

    # random sampler with replacement for training
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    rand_sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=Config.batch_size, replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.batch_size, sampler=rand_sampler)

    model_params = {
        'input_dim': x_train.shape[1],
        'selector_hdim': Config.selector_hdim,
        'predictor_hdim': Config.predictor_hdim,
        'output_dim': y_train.shape[1],
        'temp_anneal': Config.temp_anneal
    }

    net = invase_model(model_params).to(device)
    print(net)

    loss_func = ce_loss_with_reg(Config.l2, device)
    optimiser = torch.optim.Adam(net.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)

    print("Initialising training")
    for iteration in range(Config.iterations):
        train_iteration(net, train_dataloader, loss_func, optimiser, device, iteration, log_interval=100)
    print("Training complete\n")

    # evaluate performance of feature importance
    g_hat = feature_importance_score(net, x_test, device)
    importance_score = 1.*(g_hat>0.5)
    mean_tpr, std_tpr, mean_fdr, std_fdr = feature_performance_metric(g_test, importance_score)
    print("Feature importance evaluation: ")
    print(f"TPR mean: {np.round(mean_tpr,1)}%, TPR std: {np.round(std_tpr,1)}%")
    print(f"FDR mean: {np.round(mean_fdr,1)}%, FDR std: {np.round(std_fdr,1)}%")

    # evaluate performance in prediction
    y_hat = predict(net, x_test, device)
    auc, apr, acc = prediction_performance_metric(y_test, y_hat)
    print("Prediction evaluation: ")
    print(f"AUC: {np.round(auc, 3)}%, APR: {np.round(apr, 3)}%, ACC: {np.round(acc, 3)}%")

if __name__ == '__main__':
    main()