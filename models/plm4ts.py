import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.Dataset_MM as Dataset_MM

from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope
from transformers.models.bert.modeling_bert_wope import BertModel_wope
from models.embed import *

# 简单的残差解码头，用于将隐状态映射为数值
class ResHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ResHead, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x[:, :, :-1] if x.shape[-1] > self.layer1.in_features else x
        x = self.layer1(x)
        x = self.act(x)
        x = self.dropout(x)
        if residual.shape[-1] == x.shape[-1]:
            x = self.norm(x + residual)
        else:
            x = self.norm(x)
        return self.layer2(x)

class ists_plm(nn.Module):
    
    def __init__(self, opt):
        super(ists_plm, self).__init__()
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.enc_embedding = DataEmbedding_ITS_Ind_VarPrompt(self.feat_dim, self.d_model, self.feat_dim, device=opt.device, dropout=opt.dropout)

        self.gpts = nn.ModuleList()
        for i in range(2):
            gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
            bert = BertModel_wope.from_pretrained('./PLMs/bert-base-uncased', output_attentions=True, output_hidden_states=True)
            if(i==0):
                if opt.te_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_te_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.te_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_te_plmlayer]
                    self.gpts.append(bert)
            elif(i==1):
                if opt.st_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_st_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.st_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_st_plmlayer]
                    self.gpts.append(bert)
        
        if(opt.semi_freeze):
            for i in range(len(self.gpts)):
                for _, (name, param) in enumerate(self.gpts[i].named_parameters()):
                    if 'ln' in name or 'LayerNorm' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        else:
            for i in range(len(self.gpts)):
                for _, (name, param) in enumerate(self.gpts[i].named_parameters()):
                    param.requires_grad = False

        self.ln_proj = nn.LayerNorm(self.d_model)
        
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        B, L, D = observed_data.shape
        outputs, var_embedding = self.enc_embedding(observed_tp, observed_data, observed_mask) 
        outputs = self.gpts[0](inputs_embeds=outputs).last_hidden_state 
        observed_mask = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1) 
        observed_mask = torch.cat([torch.ones_like(observed_mask[:,:1]), observed_mask], dim=1) 
        n_nonmask = observed_mask.sum(dim=1) 
        outputs = (outputs * observed_mask).sum(dim=1) / n_nonmask 
        outputs = self.ln_proj(outputs.view(B, D, -1)) 
        outputs = outputs + var_embedding.squeeze()
        outputs = self.gpts[1](inputs_embeds=outputs).last_hidden_state 
        outputs = outputs.view(B, -1)
        return outputs 

class istsplm_forecast(nn.Module):
    
    def __init__(self, opt):
        super(istsplm_forecast, self).__init__()
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.input_len = opt.input_len
        self.enc_embedding = DataEmbedding_ITS_Ind_VarPrompt(self.feat_dim, self.d_model, self.feat_dim, device=opt.device, dropout=opt.dropout)

        self.gpts = nn.ModuleList()
        for i in range(2):
            gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
            bert = BertModel_wope.from_pretrained('./PLMs/bert-base-uncased', output_attentions=True, output_hidden_states=True)
            print(opt.te_model, opt.st_model)
            if(i==0):
                if opt.te_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_te_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.te_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_te_plmlayer]
                    self.gpts.append(bert)
            elif(i==1):
                if opt.st_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_st_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.st_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_st_plmlayer]
                    self.gpts.append(bert)
        
        if(opt.semi_freeze):
            print("Semi-freeze gpt")
            for i in range(len(self.gpts)):
                for _, (name, param) in enumerate(self.gpts[i].named_parameters()):
                    if 'ln' in name or 'LayerNorm' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        else:
            print("Fully-freeze gpt")
            for i in range(len(self.gpts)):
                for _, (name, param) in enumerate(self.gpts[i].named_parameters()):
                    param.requires_grad = False
                    
        self.ln_proj = nn.LayerNorm(self.d_model)

        # Autoregressive Prediction Head
        # Projects [Hidden_State + Time_Encoding] -> Value
        self.predict_decoder = ResHead(
            input_dim=opt.d_model+1, 
            hidden_dim=opt.d_model, 
            output_dim=1,
            dropout=opt.dropout if hasattr(opt, 'dropout') else 0.1
        ).to(opt.device)
    
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        """
        Autoregressive Generation
        """
        B, L, D = observed_data.shape
        B, Lp = time_steps_to_predict.shape
        
        # Initialize current sequences with history
        curr_data = observed_data
        curr_tp = observed_tp
        curr_mask = observed_mask
        
        # We will collect predictions here
        predictions = []

        # Start Autoregressive Loop
        for i in range(Lp):
            # 1. Embed current sequence
            # Note: For efficiency, one should implement KV-caching, but here we re-run for simplicity and correctness with existing embedding layers
            outputs, var_embedding = self.enc_embedding(curr_tp, curr_data, curr_mask) # (B*D, Current_L+1, d_model)
            
            # 2. Time-GPT (Intra-series)
            # Crucial: We do NOT pool here. We keep the sequence.
            outputs = self.gpts[0](inputs_embeds=outputs).last_hidden_state # (B*D, Current_L+1, d_model)
            
            # 3. Handle Dimensions for Var-GPT
            # Reshape to (B, Current_L+1, D, d_model)
            outputs = outputs.view(B, D, -1, self.d_model).permute(0, 2, 1, 3) # (B, Current_L+1, D, d_model)
            
            # Add variable embedding
            outputs = outputs + var_embedding.unsqueeze(1) # Broadcast var_embedding
            
            # Flatten time and batch for Var-GPT processing: (B * (Current_L+1), D, d_model)
            curr_len = outputs.shape[1]
            outputs_flat = outputs.reshape(B * curr_len, D, self.d_model)
            
            # 4. Var-GPT (Inter-series)
            # This allows variables to interact at every time step
            outputs_flat = self.gpts[1](inputs_embeds=outputs_flat).last_hidden_state # (B*L, D, d_model)
            
            # Reshape back: (B, Current_L+1, D, d_model)
            outputs = outputs_flat.view(B, curr_len, D, self.d_model)
            
            # 5. Get the last hidden state for prediction
            last_hidden = outputs[:, -1, :, :] # (B, D, d_model)
            
            # 6. Prepare input for decoder (Hidden + Next Time Stamp)
            next_t = time_steps_to_predict[:, i:i+1] # (B, 1)
            # Expand time to (B, D, 1)
            next_t_expanded = next_t.unsqueeze(-1).repeat(1, D, 1) # (B, D, 1)
            
            # Concat hidden state with target time
            # decoder_input: (B, D, d_model + 1)
            decoder_input = torch.cat([last_hidden, next_t_expanded], dim=-1)
            
            # 7. Predict Next Value
            pred_val = self.predict_decoder(decoder_input) # (B, D, 1)
            pred_val = pred_val.squeeze(-1) # (B, D)
            
            predictions.append(pred_val.unsqueeze(1)) # (B, 1, D)
            
            # 8. Append prediction to current sequence for next iteration
            # Update Data
            curr_data = torch.cat([curr_data, pred_val.unsqueeze(1)], dim=1) # (B, L+1, D)
            # Update Time
            curr_tp = torch.cat([curr_tp, next_t], dim=1) # (B, L+1)
            # Update Mask (Generated values are treated as observed/mask=1 for the model context)
            new_mask = torch.ones_like(pred_val.unsqueeze(1))
            curr_mask = torch.cat([curr_mask, new_mask], dim=1)

        # Concatenate all predictions
        # Output shape: (1, B, Lp, D) as expected by evaluation code
        final_output = torch.cat(predictions, dim=1).unsqueeze(0)
        return final_output


class istsplm_vector_forecast(nn.Module):
    
    def __init__(self, opt):
        super(istsplm_vector_forecast, self).__init__()
        self.seq_len = opt.max_len
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model

        self.enc_embedding = DataEmbedding_ITS_Vector(self.feat_dim, self.d_model, opt.dropout)

        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        self.predict_decoder = nn.Sequential(
			nn.Linear(opt.d_model+1, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.input_dim)
			).to(opt.device)

        
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        observed_tp = observed_tp.unsqueeze(dim=-1)
        observed_mask_agg = observed_mask.sum(dim=-1, keepdim=True).bool() 
        outputs = self.enc_embedding(observed_tp, observed_data, observed_mask, observed_mask_agg) 
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state 
        n_nonmask = observed_mask_agg.sum(dim=1) 
        outputs = (outputs * observed_mask_agg).sum(dim=1) / (n_nonmask + 1e-8) 
        B, Lp = time_steps_to_predict.size()
        time_pred = time_steps_to_predict.unsqueeze(dim = -1) 
        h = outputs.unsqueeze(dim=1).repeat(1, Lp, 1) 
        h = torch.cat([h, time_pred], dim=-1) 
        output = self.predict_decoder(h).unsqueeze(dim=0) 
        return output 

class istsplm_set_forecast(nn.Module):
    
    def __init__(self, opt):
        super(istsplm_set_forecast, self).__init__()
        self.seq_len = opt.max_len
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.plm_maxlen = 1024
        self.enc_embedding = DataEmbedding_ITS_Set(self.feat_dim, self.d_model, opt.dropout)

        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.predict_decoder = nn.Sequential(
			nn.Linear(opt.d_model+1, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.d_model),
			nn.ReLU(inplace=True),
			nn.Linear(opt.d_model, opt.input_dim)
			).to(opt.device)

    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        observed_triple, observed_mask = Dataset_MM.collate_fn_triple(observed_tp, observed_data, observed_mask)
        observed_mask = observed_mask.unsqueeze(dim=-1)
        outputs = self.enc_embedding(observed_triple, observed_mask)
        outputs = outputs[:,:self.plm_maxlen]
        observed_mask = observed_mask[:,:self.plm_maxlen]
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state 
        n_nonmask = observed_mask.sum(dim=1)  
        outputs = (outputs * observed_mask).sum(dim=1) / (n_nonmask + 1e-8) 
        B, Lp = time_steps_to_predict.size()
        time_pred = time_steps_to_predict.unsqueeze(dim = -1) 
        h = outputs.unsqueeze(dim=1).repeat(1, Lp, 1) 
        h = torch.cat([h, time_pred], dim=-1) 
        output = self.predict_decoder(h).unsqueeze(dim=0) 
        return output 

class Classifier(nn.Module):
    def __init__(self, dim, cls_dim, activate=None):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(dim, cls_dim)
    def forward(self, ENCoutput):
        ENCoutput = self.linear(ENCoutput)
        return ENCoutput
class istsplm_vector(nn.Module):
    def __init__(self, opt):
        super(istsplm_vector, self).__init__()
        self.seq_len = opt.max_len
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.enc_embedding = DataEmbedding_ITS_Vector(self.feat_dim, self.d_model, opt.dropout)
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        observed_mask_agg = observed_mask.sum(dim=-1, keepdim=True).bool() 
        outputs = self.enc_embedding(observed_tp, observed_data, observed_mask, observed_mask_agg) 
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state 
        n_nonmask = observed_mask_agg.sum(dim=1)  
        outputs = (outputs * observed_mask_agg).sum(dim=1) / (n_nonmask + 1e-8) 
        return outputs
class istsplm_set(nn.Module):
    def __init__(self, opt):
        super(istsplm_set, self).__init__()
        self.seq_len = opt.max_len
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.plm_maxlen = 1024
        self.enc_embedding = DataEmbedding_ITS_Set(self.feat_dim, self.d_model, opt.dropout)
        self.gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:opt.n_te_plmlayer]
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        observed_tp = observed_tp.squeeze(dim=-1)
        observed_triple, observed_mask = Dataset_MM.collate_fn_triple(observed_tp, observed_data, observed_mask)
        observed_mask = observed_mask.unsqueeze(dim=-1)
        outputs = self.enc_embedding(observed_triple, observed_mask) 
        outputs = outputs[:,:self.plm_maxlen]
        observed_mask = observed_mask[:,:self.plm_maxlen]
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state 
        n_nonmask = observed_mask.sum(dim=1)  
        outputs = (outputs * observed_mask).sum(dim=1) / (n_nonmask + 1e-8) 
        return outputs
