import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class PreferenceLearner(nn.Module):

    def __init__(self, gpu, input_size, num_hidden):
    
        super(PreferenceLearner, self).__init__()

        self.device = torch.device("cuda" if gpu >= 0 else "cpu")
        
        self.fc1 = nn.Linear(in_features=input_size, out_features=num_hidden).to(self.device)
        self.fc2 = nn.Linear(in_features=num_hidden, out_features=num_hidden).to(self.device)
        self.fc3 = nn.Linear(in_features=num_hidden, out_features=1).to(self.device)

        self.dropout_p = 0.5

        self.relu = nn.ReLU().to(self.device)
        self.tanh = nn.Tanh().to(self.device)
        
        self.fc1.bias.data.mul_(0.0)
        self.fc2.bias.data.mul_(0.0)
        self.fc3.bias.data.mul_(0.0)


    def forward(self, x, use_dropout, regen_mask=True):
    
        x = self.relu(self.fc1(x))
        
        #if self.training and use_dropout:
        #    if regen_mask:
        #        self.dropout_mask_1 = torch.bernoulli(torch.ones_like(x).mul(1.0 - self.dropout_p)).mul(1.0 / (1.0 - self.dropout_p))
        #    x = x.mul(self.dropout_mask_1.detach())
        
        x = self.relu(self.fc2(x))
        
        if self.training and use_dropout:
            if regen_mask:
                self.dropout_mask_2 = torch.bernoulli(torch.ones_like(x).mul(1.0 - self.dropout_p)).mul(1.0 / (1.0 - self.dropout_p))
            x = x.mul(self.dropout_mask_2.detach())
            
        x = self.fc3(x)
        return x


    def normalise_output(self, results_to_norm):
    
        out = self.forward(results_to_norm, use_dropout=False).squeeze().detach().numpy()

        mean = np.mean(out)
        std_dev = np.std(out, ddof=1)

        self.fc3.weight.data.div_(std_dev)
        self.fc3.bias.data.div_(std_dev)
        self.fc3.bias.data.sub_(mean / std_dev)
        
    
    def get_grad_norm(self):
    
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
        
        
    def predict_preference(self, match_results, use_dropout):
    
        squeeze_dim = len(match_results.shape) - 2 # To allow batching
        out_a = self.forward(match_results.narrow(squeeze_dim, 0, 1).squeeze(), use_dropout)
        out_b = self.forward(match_results.narrow(squeeze_dim, 1, 1).squeeze(), use_dropout, regen_mask=False)
        
        #x = self.tanh(out_a - out_b).add(1.0).div(2.0)
        #return x
        
        if len(out_a.shape) == 1:
            out_merge = torch.cat((out_a, out_b), dim=0)
            softmax_out = F.softmax(out_merge, dim=0)
            x = softmax_out.narrow(0, 0, 1) #.mul(2.0).sub(1.0)
        else:
            out_merge = torch.cat((out_a, out_b), dim=1)
            softmax_out = F.softmax(out_merge, dim=1)
            x = softmax_out.narrow(1, 0, 1) #.mul(2.0).sub(1.0)
        
        return x


    def get_uncertainty(self, match_results, top_n, num_fwd_passes):
    
        state_backup = self.training
        self.train()
        
        estimates = np.zeros([num_fwd_passes, match_results.size(0)], dtype = float)
        
        for i in range(0, num_fwd_passes):
            estimates[i] = self.predict_preference(match_results, use_dropout=True).detach().numpy()[:, 0]
            
        self.training = state_backup
        
        std_dev = np.std(estimates, axis=0, ddof=1)
        return np.argpartition(std_dev, -top_n)[-top_n:]


    def save_model(self, output_dir):
    
        fc1_w_np = self.fc1.weight.detach().numpy()
        fc2_w_np = self.fc2.weight.detach().numpy()
        fc3_w_np = self.fc3.weight.detach().numpy()

        fc1_b_np = self.fc1.bias.detach().numpy()
        fc2_b_np = self.fc2.bias.detach().numpy()
        fc3_b_np = self.fc3.bias.detach().numpy()

        pd.DataFrame(fc1_w_np).to_csv(output_dir + '/fc1_w.csv')
        pd.DataFrame(fc2_w_np).to_csv(output_dir + '/fc2_w.csv')
        pd.DataFrame(fc3_w_np).to_csv(output_dir + '/fc3_w.csv')

        pd.DataFrame(fc1_b_np).to_csv(output_dir + '/fc1_b.csv')
        pd.DataFrame(fc2_b_np).to_csv(output_dir + '/fc2_b.csv')
        pd.DataFrame(fc3_b_np).to_csv(output_dir + '/fc3_b.csv')
        
        
    def load_model(self, load_dir):
    
        fc1_w_np = np.loadtxt(load_dir + '/fc1_w.csv', delimiter=",", skiprows=1)
        fc2_w_np = np.loadtxt(load_dir + '/fc2_w.csv', delimiter=",", skiprows=1)
        fc3_w_np = np.loadtxt(load_dir + '/fc3_w.csv', delimiter=",", skiprows=1)
        
        fc1_b_np = np.loadtxt(load_dir + '/fc1_b.csv', delimiter=",", skiprows=1)
        fc2_b_np = np.loadtxt(load_dir + '/fc2_b.csv', delimiter=",", skiprows=1)
        fc3_b_np = np.loadtxt(load_dir + '/fc3_b.csv', delimiter=",", skiprows=1)
        
        fc1_w_np = fc1_w_np[:, 1:]
        fc2_w_np = fc2_w_np[:, 1:]
        fc3_w_np = fc3_w_np[1:]
        
        fc1_b_np = fc1_b_np[:, 1:]
        fc2_b_np = fc2_b_np[:, 1:]
        fc3_b_np = fc3_b_np[1:]
        
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(fc1_w_np))
            self.fc2.weight.copy_(torch.from_numpy(fc2_w_np))
            self.fc3.weight.copy_(torch.from_numpy(fc3_w_np))
            self.fc1.bias.copy_(torch.from_numpy(fc1_b_np).squeeze())
            self.fc2.bias.copy_(torch.from_numpy(fc2_b_np).squeeze())
            self.fc3.bias.copy_(torch.from_numpy(fc3_b_np))

