import argparse  
import os
import torch
from exp.exp_long_term_forecasting_STW import Exp_Long_Term_Forecast  
from utils.print_args import print_args 
import random
import numpy as np



if __name__ == '__main__':
    fix_seed = 2021 
    random.seed(fix_seed) 
    torch.manual_seed(fix_seed) 
    np.random.seed(fix_seed) 
 
    parser = argparse.ArgumentParser(description='DWFN')
    
    ######################################## 
    # Basic Configuration Parameters 
    ######################################## 
    parser.add_argument('--task_name',  type=str, default='long_term_forecast',
                        help='Task name, options: [long_term_forecast]')
    parser.add_argument('--is_training',  type=int, default=1, help='Status: 1 for training, 0 for testing')
    parser.add_argument('--model',  type=str, default='DWTN', help='Model name')
    parser.add_argument('--embed',  type=str, default='timeF',
                        help='Time feature encoding method: [timeF, fixed, learned]')
    ######################################## 
    # Data Loading Configuration 
    ######################################## 
    
    parser.add_argument('--features',  type=str, default='M',
                        help='Forecasting task type: [M, S, MS] M: multivariate -> multivariate, S: univariate -> univariate, MS: multivariate -> univariate')
    parser.add_argument('--target',  type=str, default='1', help='Target feature in S or MS tasks')
    parser.add_argument('--freq',  type=str, default='h',
                        help='Frequency encoding for time features: [s, t, h, d, b, w, m] or detailed like 15min')
    parser.add_argument('--checkpoints',  type=str, default='./checkpoints/', help='Path to save model checkpoints')
 
    ######################################## 
    # Forecasting Task Parameters 
    ######################################## 
    parser.add_argument('--seq_len',  type=int, default=96, help='Input sequence length')
    parser.add_argument('--label_len',  type=int, default=48, help='Start token length')
    parser.add_argument('--seasonal_patterns',  type=str, default='Monthly', help='M4 dataset subset')
    parser.add_argument('--inverse',  action='store_true', default=False, help='Whether to inverse output data')
 
    parser.add_argument('--channel_independence',  type=int, default=0,
                        help='Channel independence setting (1: dependent 0: independent)')
 
    parser.add_argument('--dropout',  type=float, default=0, help='Dropout rate')
 
    ######################################## 
    # Optimization & Training Parameters 
    ######################################## 
    parser.add_argument('--num_workers',  type=int, default=0, help='Number of data loading threads')
    parser.add_argument('--itr',  type=int, default=1, help='Number of experiment repetitions')
    parser.add_argument('--patience',  type=int, default=5, help='Early stopping patience rounds')
    parser.add_argument('--des',  type=str, default='test', help='Experiment description')
    parser.add_argument('--loss',  type=str, default='MSE', help='Loss function')
    parser.add_argument('--lradj',  type=str, default='type1', help='Learning rate adjustment strategy')
    parser.add_argument('--use_amp',  action='store_true', default=False, help='Whether to use mixed precision training')
 
    ######################################## 
    # GPU Configuration 
    ######################################## 
    parser.add_argument('--use_gpu',  type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--gpu',  type=int, default=0, help='GPU ID to use')
    parser.add_argument('--use_multi_gpu',  action='store_true', default=False, help='Whether to use multiple GPUs')
    parser.add_argument('--devices',  type=str, default='0,1,2,3', help='Device IDs of multiple GPUs')
 
    ############################################ 
 
    # =========== Model Architecture Parameters =========== 
    parser.add_argument('--hidden_size',   type=int, default=96, help='Hidden layer dimension')
    parser.add_argument('--mlp_dropout',   type=float, default=0.1, help='Dropout rate for MLP modules')
    parser.add_argument('--embed_size',  default=128, type=int, help='Intermediate dimension of wavelet coefficient MLP')
    # =========== Wavelet Transform Parameters =========== 
    parser.add_argument('--swt_level',   type=int, default=1, 
                        help='SWT decomposition levels (require 2^level <= seq_len)')
    parser.add_argument('--wavelet',   type=str, default='db3', 
                        help='Wavelet basis type, options: db1-40/sym2-20/coif1-5 etc.')
    
    # =========== Loss Calculation Parameters =========== 
    parser.add_argument('--time_lambda',   type=float, default=0.5,
                        help='Weight coefficient for time domain MSE loss')
    parser.add_argument('--energy_lambda',   type=float, default=1,
                        help='Weight coefficient for energy conservation loss')
    parser.add_argument('--auxi_lambda',   type=float, default=1,
                        help='Weight coefficient for frequency domain loss')
    parser.add_argument('--cA_lambda',   type=float, nargs='+', default=[0.5, 0.3],
                        help='MAE weights for each level\'s approximation coefficients (length should equal swt_level)')
    parser.add_argument('--cD_lambda',   type=float, nargs='+', default=[0.2, 0.1],
                        help='MAE weights for each level\'s detail coefficients (length should equal swt_level)')
    
    parser.add_argument('--pred_len',  type=int, default=96, help='Prediction sequence length')
    parser.add_argument('--batch_size',  type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate',  type=float, default=0.006, help='Learning rate')
    parser.add_argument('--train_epochs',  type=int, default=10, help='Training epochs')
    
    parser.add_argument('--model_id',  type=str, default='ETTh1', help='Model ID')
    parser.add_argument('--data',  type=str, default='ETTh1', help='Dataset type')
    parser.add_argument('--root_path',  type=str, default='./all_six_datasets/ETT-small', help='Root path of data file')
    parser.add_argument('--data_path',  type=str, default='ETTh1.csv',  help='Data file name')
    parser.add_argument('--enc_in',  type=int, default=7, help='Encoder input dimension')
    
    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast
    
    if args.is_training == 1:
        for ii in range(args.itr):
            setting = '{}_{}_{}_sl{}_pl{}_embed{}_hidden{}_bs{}_lr{}_drop{}_{}'.format(
                args.model_id, args.model, args.data, args.seq_len, args.pred_len,
                args.embed_size, args.hidden_size, args.batch_size, args.learning_rate,
                args.dropout, ii)
            
            exp = Exp(args)  
            
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_pl{}_embed{}_hidden{}_bs{}_lr{}_drop{}_{}'.format(
            args.model_id, args.model, args.data, args.seq_len, args.pred_len,
            args.embed_size, args.hidden_size, args.batch_size, args.learning_rate,
            args.dropout, ii)
        
        exp = Exp(args)  
        
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        
        torch.cuda.empty_cache()