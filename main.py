from data_generation import generate_dataset
from model import invase_model, ce_loss_with_reg
from trainer import train_one_iter
from evaluator import feature_importance_score, predict
from utils import feature_performance_metric, prediction_performance_metric
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Config:
    train_no = 10000
    test_no = 10000
    dim = 11
    data_type = "syn6"
    batch_size = 100
    selector_hdim = 100
    predictor_hdim = 200
    weight_decay = 1e-4
    lr = 1e-3
    l2 = 1e-3
    temp_anneal = 1e-3
    iterations = 2000
    save_model = True
    patience = 200

def main():
    tpr_cv = []
    fdr_cv = []
    auc_cv = []
    for l2 in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        net = invase_model(model_params).to(device)

        # weight loss function by class support
        class_weight = [1/(np.sum(y_train[:, 0]/y_train.shape[0])), 1/(np.sum(y_train[:, 1]/y_train.shape[0]))]
        loss_func = ce_loss_with_reg(l2, Config.weight_decay, device, class_weight)
        optimiser = torch.optim.Adam(net.parameters(), lr=Config.lr)
        # optimiser = torch.optim.Adam(net.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
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

        # load best model
        net.load_state_dict(torch.load(save_model_dir, map_location=torch.device(device)))
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

        tpr_cv.append(mean_tpr)
        fdr_cv.append(mean_fdr)
        auc_cv.append(auc)
    print(tpr_cv)
    print(fdr_cv)
    print(auc_cv)

if __name__ == '__main__':
    main()