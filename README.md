# "INVASE+: INSTANCE-WISE VARIABLE SELECTION USING PATH-BASED DERIVATIVES"

Extension to INVASE model proposed by the authors below:

Paper: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
       "INVASE: Instance-wise Variable Selection using Neural Networks," 
       International Conference on Learning Representations (ICLR), 2019.
       (https://openreview.net/forum?id=BJg_roAcK7)
Github: https://github.com/jsyoon0823/INVASE

INVASE+ extends INVASE by using path-based derivatives to allow backpropagation through subset sampling. Additionally, the selector and predictor networks are trained in an embedded fashion. Please see `invase_plus.pdf` for more information.

## Stages of the INVASE+ framework:
-   Generate synthetic dataset (6 synthetic datasets)
-   Train INVASE+
-   Evaluate INVASE+ for instance-wise feature selection
-   Evaluate INVASE+ for prediction

## Running and Configuring Model:
Training and evaluation of the model are actioned through `main.py`. At run-time, `main` reads from a model configuration set in `config.py`. The main config settings are:
- data_type: synthetic data type ("syn{1-6}")
- dim: number of dimensions
- train_no: number of samples for training set
- test_no: number of samples for test set
- selector_hdim: hidden layer dimensions for selector network (MLP)
- predictor_hdim: hidden layer dimensions for predictor network (MLP)
- lr: learning rate
- weight_decay: weight regularisation (l2)
- l2: feature probability/importance regularisation (l2)
- temp_anneal: temperature annealing factor
- iterations: number of iterations to train
- batch_size: mini-batch size
- patience: for early stopping
- save_model: whether to save model after training

## Referencing:
Please cite authors and work of original INVASE paper. 