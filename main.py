'''
Main logic.
'''

import torch
import numpy as np
from data_generation import generate_dataset
from model import invase_plus, ce_loss_with_reg
from trainer import train_one_iter
from evaluator import feature_importance_score, predict
from utils import feature_performance_metric, prediction_performance_metric
from config import Config
from torch.utils.tensorboard import SummaryWriter

def main():
    '''
    Main logic for program. Trains the model and evaluates discovery and prediction performance.
    '''
    device='cpu'
    print(f"Training model on: {device}")
    writer = SummaryWriter(comment=f"_{Config.data_type}_selector_training")

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
    net = invase_plus(model_params).to(device)

    # weight loss function by class support
    class_weight = [1/(np.sum(y_train[:, 0]/y_train.shape[0])), 1/(np.sum(y_train[:, 1]/y_train.shape[0]))]
    loss_func = ce_loss_with_reg(Config.l2, Config.weight_decay, device, class_weight)
    optimiser = torch.optim.Adam(net.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=500, gamma=0.5)

    if Config.save_model:
        save_model_dir = f"trained_models/{Config.data_type}_invase_net_plus.pt"

    best_loss = float("inf")
    early_stopping_iters = 0
    print("Initialising training")
    for iteration in range(Config.iterations):
        loss, acc = train_one_iter(net, train_dataloader, loss_func, optimiser, device, iteration, log_interval=100)
        scheduler.step()
        writer.add_scalar("Loss/train", loss, iteration)
        writer.add_scalar("Acc/train", acc, iteration)
        if loss < best_loss:
            best_loss, best_acc = loss, acc
            print(f"Loss/accuracy improved: {best_loss:.4f}/{acc:.2f}%, saving model...")
            if Config.save_model:
                torch.save(net.state_dict(), save_model_dir)
            early_stopping_iters = 0
        else:
            early_stopping_iters += 1
    
        if early_stopping_iters == Config.patience:
            print(f"Early stopping after iteration {iteration}")
            break

    print("Training complete\n")

    # evaluate performance of feature importance
    net.load_state_dict(torch.load(save_model_dir, map_location=torch.device(device)))
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