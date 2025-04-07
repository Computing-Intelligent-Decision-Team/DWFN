from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
#from utils.metrics_torch import create_metric_collector, metric_torch
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch)
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pywt
import itertools
warnings.filterwarnings('ignore')
from typing import List, Tuple  


class WaveletLossCalculator:
    def __init__(self, args, device, wavelet='db3', level=2):
        self.args  = args
        self.criterion  = nn.MSELoss()   
        self.mae  = nn.L1Loss()         
        self.energy_loss  = nn.MSELoss()  
        self.device  = torch.device(device) 
        self.swt_level  = args.swt_level  
        self.wavelet  = args.wavelet 
        
        self.time_lambda  = args.time_lambda     
        self.cA_lambda = args.cA_lambda         
        self.cD_lambda = args.cD_lambda         
        self.energy_lambda  = args.energy_lambda  

 
    def _get_swt_coeffs(self, x):
        with torch.no_grad():  
            x_np = x.detach().cpu().numpy() 
    
        batch, seq_len, channels = x.shape  
        all_coeffs = []
    
        for b, ch in itertools.product(range(batch),  range(channels)):
            signal = x_np[b, :, ch]
            coeffs = pywt.swt(signal,  self.wavelet,  level=self.swt_level) 
            all_coeffs.append(coeffs) 
    
        swt_coeffs = []
        for lev in range(self.swt_level): 
            cA_list, cD_list = [], []
            for idx in range(len(all_coeffs)):
                cA_lev = all_coeffs[idx][lev][0]
                cD_lev = all_coeffs[idx][lev][1]
                cA_list.append(cA_lev) 
                cD_list.append(cD_lev) 
    
            cA_tensor = torch.tensor(np.stack(cA_list),  dtype=torch.float32, 
                                device=self.device).requires_grad_(x.requires_grad) 
            cD_tensor = torch.tensor(np.stack(cD_list),  dtype=torch.float32, 
                                device=self.device).requires_grad_(x.requires_grad) 
            
            cA_tensor = cA_tensor.view(batch,  channels, -1).transpose(1, 2)
            cD_tensor = cD_tensor.view(batch,  channels, -1).transpose(1, 2)
            
            swt_coeffs.append((cA_tensor,  cD_tensor))
        
        return swt_coeffs 
 
    def _compute_coeff_loss(self, pred, target):
        pred_coeffs = self._get_swt_coeffs(pred)
        target_coeffs = self._get_swt_coeffs(target)
        
        total_loss = 0 
        for lev in range(self.swt_level): 
            decay_factor = 1.0 / (2 ** (self.swt_level  - lev - 1))
            
            pred_cA, pred_cD = pred_coeffs[lev]
            target_cA, target_cD = target_coeffs[lev]
            
            cA_loss = self.mae(pred_cA,  target_cA) * self.cA_lambda[lev] * decay_factor 
            cD_loss = self.mae(pred_cD,  target_cD) * self.cD_lambda[lev] * decay_factor 
            
            total_loss += cA_loss + cD_loss 
        
        return total_loss 
 
    def _energy_conservation(self, pred, target):
        pred_coeffs = self._get_swt_coeffs(pred)
        target_coeffs = self._get_swt_coeffs(target)
        
        pred_energy = []
        target_energy = []
        for lev in range(self.swt_level): 
            pred_energy_lev = (pred_coeffs[lev][0].square().mean() + 
                              pred_coeffs[lev][1].square().mean())
            target_energy_lev = (target_coeffs[lev][0].square().mean() + 
                                target_coeffs[lev][1].square().mean())
            
            pred_energy.append(pred_energy_lev) 
            target_energy.append(target_energy_lev) 
        
        return self.criterion(torch.stack(pred_energy),  torch.stack(target_energy)) 
    
    def _compute_spectral_loss(self, outputs, targets):
        output_fft = torch.fft.rfft(outputs,  dim=1)
        target_fft = torch.fft.rfft(targets,  dim=1)
        
        real_diff = (output_fft.real  - target_fft.real) 
        imag_diff = (output_fft.imag  - target_fft.imag) 
        
        real_loss = real_diff.abs().mean() 
        imag_loss = imag_diff.abs().mean() 

        return real_loss + imag_loss
 
    def compute_total_loss(self, outputs, targets):
        time_loss = self.time_lambda  * self.criterion(outputs,  targets)
        
        coeff_loss = self._compute_coeff_loss(outputs, targets)
        energy_loss = self.energy_lambda  * self._energy_conservation(outputs, targets)
        
        spectral_loss = self.args.auxi_lambda  * self._compute_spectral_loss(outputs, targets)
        total_loss = time_loss + coeff_loss + energy_loss + spectral_loss
        return total_loss 


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.pred_len = args.pred_len
        self.loss_calculator  = WaveletLossCalculator(args, self.device) 

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    #if self.args.output_attention:
                    #    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    #else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                #loss = criterion(pred, true)
                #total_loss.append(loss)
                loss = criterion(pred, true)
                total_loss.append(loss.item())   
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints,  setting)
        if not os.path.exists(path): 
            os.makedirs(path) 
    
        time_now = time.time() 
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience,  verbose=True)
        model_optim = self._select_optimizer()
        
        print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        print('Model {} : params: {:4f}M'.format(self.model._get_name(), para * 4 / 1024 / 1024))
        
        loss_calculator = WaveletLossCalculator(
            args=self.args, 
            device=self.device, 
        )
    
        if self.args.use_amp: 
            scaler = torch.cuda.amp.GradScaler() 
    
        for epoch in range(self.args.train_epochs): 
            iter_count = 0 
            train_loss = []
            self.model.train() 
            epoch_time = time.time() 
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1 
                model_optim.zero_grad() 
                
                batch_x = batch_x.float().to(self.device) 
                batch_y = batch_y.float().to(self.device) 
                batch_x_mark = batch_x_mark.float().to(self.device) 
                batch_y_mark = batch_y_mark.float().to(self.device) 
                
                dec_inp = torch.cat([ 
                    batch_y[:, :self.args.label_len,  :],
                    torch.zeros_like(batch_y[:,  -self.args.pred_len:,  :])
                ], dim=1).float().to(self.device) 
    
                if self.args.use_amp: 
                    with torch.cuda.amp.autocast(): 
                        outputs = self.model(batch_x,  batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention: 
                            outputs = outputs[0]
                else:
                    outputs = self.model(batch_x,  batch_x_mark, dec_inp, batch_y_mark)
    
                f_dim = -1 if self.args.features  == 'MS' else 0 
                outputs = outputs[:, -self.args.pred_len:,  f_dim:]
                targets = batch_y[:, -self.args.pred_len:,  f_dim:].to(self.device) 
                loss = loss_calculator.compute_total_loss(outputs,  targets)
    
                if self.args.use_amp: 
                    scaler.scale(loss).backward() 
                    scaler.step(model_optim) 
                    scaler.update() 
                else:
                    loss.backward() 
                    model_optim.step() 
    
                train_loss.append(loss.item()) 
                if (i + 1) % 100 == 0:
                    speed = (time.time()  - time_now) / iter_count 
                    left_time = speed * ((self.args.train_epochs  - epoch) * train_steps - i)
                    print(f"\tEpoch: {epoch+1} | Iter: {i+1} | Loss: {loss.item():.4f}  | Speed: {speed:.2f}s/iter | ETA: {left_time:.2f}s")
                    iter_count = 0 
                    time_now = time.time() 
    
            vali_loss = self.vali(vali_data,  vali_loader, loss_calculator.criterion) 
            test_loss = self.vali(test_data,  test_loader, loss_calculator.criterion) 
            print(f"Epoch: {epoch+1} | Train Loss: {np.mean(train_loss):.4f}  | Val Loss: {vali_loss:.4f} | Test Loss: {test_loss:.4f}")
            
            early_stopping(vali_loss, self.model,  path)
            if early_stopping.early_stop: 
                print("Early stopping triggered")
                break 
                
            adjust_learning_rate(model_optim, epoch+1, self.args) 
    
        self.model.load_state_dict(torch.load(os.path.join(path,  "checkpoint.pth"))) 
        return self.model  

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    if i % 100 == 0 :
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
