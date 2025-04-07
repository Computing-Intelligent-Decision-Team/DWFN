import torch 
import torch.nn  as nn 
from layers.RevIN import RevIN 
import numpy as np
import random
import pywt
import itertools

class SWTLayer(torch.autograd.Function):   
    @staticmethod 
    def forward(ctx, x, wavelet, level):
        batch, channels, seq_len = x.shape   
        x_np = x.detach().cpu().numpy() 
 
        merged_coeffs = []
        for lev in range(level):
            cA_lev, cD_lev = [], []
            for b, ch in itertools.product(range(batch),  range(channels)):
                signal = x_np[b, ch, :]
                coeffs = pywt.swt(signal,  wavelet, level=level)  
                assert len(coeffs[lev]) == 2, f"层级{lev}系数异常" 
                cA, cD = coeffs[lev]
                cA_lev.append(cA) 
                cD_lev.append(cD) 
            
            cA_array = np.stack(cA_lev).reshape(batch,  channels, -1)
            cD_array = np.stack(cD_lev).reshape(batch,  channels, -1)
            merged_coeffs.append((torch.tensor(cA_array,  device=x.device), 
                                torch.tensor(cD_array,  device=x.device))) 
        return merged_coeffs 
    
    
class ISWTLayer(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx, coeffs, wavelet):
        np_coeffs = [(cA.detach().cpu().numpy(),  
                     cD.detach().cpu().numpy())  for cA, cD in coeffs]
        
        reconstructed_np = pywt.iswt(np_coeffs,  wavelet)
        
        ctx.wavelet  = wavelet 
        ctx.shape  = coeffs[0][0].shape  
        
        reconstructed = torch.tensor(reconstructed_np,  
                                   dtype=coeffs[0][0].dtype,
                                   device=coeffs[0][0].device)
        return reconstructed 
 
    @staticmethod 
    def backward(ctx, grad_output):
        wavelet = ctx.wavelet  
        original_shape = ctx.shape  
        
        grad_coeffs = []
        grad_np = grad_output.detach().cpu().numpy() 
        
        for _ in range(len(ctx.shape)): 
            grad_coeff = pywt.swt(grad_np,  wavelet, level=1)
            grad_coeffs.append( 
                (torch.tensor(grad_coeff[0][0],  device=grad_output.device), 
                 torch.tensor(grad_coeff[0][1],  device=grad_output.device)) 
            )
        
        return grad_coeffs, None 

class GatedMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.gate_proj  = nn.Linear(input_dim, input_dim * 2)
        self.dropout  = nn.Dropout(dropout)
        self.layer_norm  = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        residual = x 
        gates = self.gate_proj(x) 
        gate_a, gate_b = gates.chunk(2,  dim=-1)
        output = gate_a * torch.sigmoid(gate_b) 
        return self.layer_norm(self.dropout(output)  + residual)

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 保留原始参数结构 
        self.seq_len  = configs.seq_len   
        self.pred_len  = configs.pred_len   
        self.enc_in  = configs.enc_in   
        self.revin_layer  = RevIN(configs.enc_in,  affine=True, subtract_last=False)

        self.hidden_size  = configs.hidden_size   
        self.swt_level  = configs.swt_level        
        self.wavelet  = configs.wavelet           
        
        self.mlp_coeffs  = nn.ModuleList()
        
        self.mlp_hidden_dim = configs.embed_size
        
        self.mlp_dropout = configs.mlp_dropout
        
        self.wavelet_dim  = self.seq_len   
        for _ in range(self.swt_level  * 2):
            self.mlp_coeffs.append( 
                self._build_MLP_block(
                    input_dim=self.wavelet_dim, 
                    hidden_dim=self.mlp_hidden_dim, 
                    mlp_dropout=self.mlp_dropout  
                )
            )
        
        self.fc  = self._build_head() 
        
    def _build_MLP_block(self, input_dim, hidden_dim, mlp_dropout):
        return GatedMLPBlock(input_dim, hidden_dim, mlp_dropout)
     
     
    def _build_head(self):
        return nn.Sequential(
            nn.Linear(self.seq_len,  self.hidden_size*4),   
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),  
            nn.Linear(self.hidden_size*4,  self.hidden_size), 
            nn.LayerNorm(self.hidden_size), 
            nn.Linear(self.hidden_size,  self.pred_len)  
        )
        
    def fre_process(self, x):
        coeffs = SWTLayer.apply(x,  self.wavelet,  self.swt_level) 
        processed_coeffs = []
        for lev, (cA, cD) in enumerate(coeffs):
            cA_flat = cA.reshape(-1,  cA.shape[2])  
            cD_flat = cD.reshape(-1,  cD.shape[2])  
            
            mlp_a = self.mlp_coeffs[2*lev] 
            mlp_d = self.mlp_coeffs[2*lev+1] 
            
            cA_proc = mlp_a(cA_flat) 
            cD_proc = mlp_d(cD_flat)
            
            cA_restored = cA_proc.view( 
                x.shape[0],  x.shape[1],  -1 
            )
            cD_restored = cD_proc.view( 
                x.shape[0],  x.shape[1],  -1 
            )
            processed_coeffs.append((cA_restored,  cA_restored))

        return ISWTLayer.apply(processed_coeffs,  self.wavelet) 
    
    
    def forward(self, x, *args):
        z = self.revin_layer(x,  'norm').permute(0, 2, 1)
        z = self.fre_process(z) 
        z = self.fc(z)
        return self.revin_layer(z.permute(0,  2, 1), 'denorm')