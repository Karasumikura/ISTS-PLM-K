import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.Dataset_MM as Dataset_MM

from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope
from transformers.models.bert.modeling_bert_wope import BertModel_wope
from models.embed import *

# 改进的解码头
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

        self.act = F.gelu
        self.dropout = nn.Dropout(p=opt.dropout)
        self.ln_proj = nn.LayerNorm(self.d_model)
        
    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        """ 
        observed_tp: (B, L, D)
        observed_data: (B, L, D) tensor containing the observed values.
        observed_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
    
        B, L, D = observed_data.shape
        
        outputs, var_embedding = self.enc_embedding(observed_tp, observed_data, observed_mask) # (B*D, L+1, d_model)
        outputs = self.gpts[0](inputs_embeds=outputs).last_hidden_state # (B*D, L+1, d_model)
        observed_mask = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1) # (B*D, L, 1)
        observed_mask = torch.cat([torch.ones_like(observed_mask[:,:1]), observed_mask], dim=1) # (B*D, L+1, 1)
        n_nonmask = observed_mask.sum(dim=1)  # (B*D, 1)
        outputs = (outputs * observed_mask).sum(dim=1) / n_nonmask # (B*D, d_model)
        outputs = self.ln_proj(outputs.view(B, D, -1))  # (B*D, (L+1)*d_model)
        outputs = outputs + var_embedding.squeeze()
        outputs = self.gpts[1](inputs_embeds=outputs).last_hidden_state # (B, D, d_model)
        outputs = outputs.view(B, -1)
        return outputs #(B, D*d_model)
    
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
        """ 
        observed_tp: (B, L, 1)
        observed_data: (B, L, D) tensor containing the observed values.
        observed_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """

        observed_mask_agg = observed_mask.sum(dim=-1, keepdim=True).bool() # (B, L, 1)
        outputs = self.enc_embedding(observed_tp, observed_data, observed_mask, observed_mask_agg) # (B, L, d_model)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # (B, L, d_model)
        
        n_nonmask = observed_mask_agg.sum(dim=1)  # (B, 1)
        outputs = (outputs * observed_mask_agg).sum(dim=1) / (n_nonmask + 1e-8) # (B, d_model)
        
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
        """ 
        observed_tp: (B, L, 1)
        observed_data: (B, L, D) tensor containing the observed values.
        observed_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        
        observed_tp = observed_tp.squeeze(dim=-1)
        observed_triple, observed_mask = Dataset_MM.collate_fn_triple(observed_tp, observed_data, observed_mask)

        observed_mask = observed_mask.unsqueeze(dim=-1)
        outputs = self.enc_embedding(observed_triple, observed_mask) # (B, L, d_model)
        outputs = outputs[:,:self.plm_maxlen]
        observed_mask = observed_mask[:,:self.plm_maxlen]

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # (B, L, d_model)
        
        n_nonmask = observed_mask.sum(dim=1)  # (B, 1)
        outputs = (outputs * observed_mask).sum(dim=1) / (n_nonmask + 1e-8) # (B, d_model)
        
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
        Autoregressive Forecasting
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
            # Note: For optimal efficiency, one should implement KV-caching.
            # Here we re-run forward pass for simplicity and stability with current architecture.
            outputs, var_embedding = self.enc_embedding(curr_tp, curr_data, curr_mask) # (B*D, L_curr+1, d_model)
            
            # 2. Time-GPT (Intra-series)
            outputs = self.gpts[0](inputs_embeds=outputs).last_hidden_state # (B*D, L_curr+1, d_model)
            
            # 3. Reshape for Var-GPT
            # outputs: (B*D, L_curr+1, d_model) -> (B, D, L_curr+1, d_model) -> (B, L_curr+1, D, d_model)
            outputs = outputs.view(B, D, -1, self.d_model).permute(0, 2, 1, 3)
            
            # Add var embedding (B, 1, D, d_model)
            # var_embedding from enc_embedding is (B, 1, D, d_model) in our context?
            # Actually enc_embedding returns (..., vars_prompt) where vars_prompt is (B, 1, D, d_model)
            # So simple addition works
            outputs = outputs + var_embedding # (B, L_curr+1, D, d_model)
            
            # 4. Var-GPT (Inter-series)
            # Flatten B and L dims: (B * (L_curr+1), D, d_model)
            outputs_flat = outputs.flatten(0, 1) 
            
            outputs_flat = self.gpts[1](inputs_embeds=outputs_flat).last_hidden_state
            
            # Reshape back: (B, L_curr+1, D, d_model)
            outputs_final = outputs_flat.view(B, -1, D, self.d_model)
            
            # Get last time step hidden state: (B, D, d_model)
            last_hidden = outputs_final[:, -1, :, :]
            
            # 5. Predict
            # next_t: (B, 1)
            next_t = time_steps_to_predict[:, i:i+1]
            
            # Expand next_t to match (B, D, 1) for decoder input
            next_t_expanded = next_t.unsqueeze(-1).repeat(1, D, 1) # (B, D, 1)
            
            decoder_input = torch.cat([last_hidden, next_t_expanded], dim=-1) # (B, D, d_model+1)
            
            pred_val = self.predict_decoder(decoder_input).squeeze(-1) # (B, D)
            
            predictions.append(pred_val.unsqueeze(1))
            
            # 6. Update History
            # curr_data: (B, L, D) -> (B, L+1, D)
            curr_data = torch.cat([curr_data, pred_val.unsqueeze(1)], dim=1)
            
            # curr_mask: (B, L, D) -> (B, L+1, D)
            curr_mask = torch.cat([curr_mask, torch.ones_like(pred_val.unsqueeze(1))], dim=1)
            
            # curr_tp: (B, L, D) -> (B, L+1, D)
            # [FIXED] Expand next_t for cat: (B, 1) -> (B, 1, D)
            next_t_for_cat = next_t.unsqueeze(-1).repeat(1, 1, D) # (B, 1, D)
            curr_tp = torch.cat([curr_tp, next_t_for_cat], dim=1)

        # Concatenate all predictions
        final_output = torch.cat(predictions, dim=1).unsqueeze(0) # (1, B, Lp, D)
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
        """ 
        combined_tt: (B, L)
        combined_vals: (B, L, D) tensor containing the observed values.
        combined_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
    
        observed_tp = observed_tp.unsqueeze(dim=-1)
        observed_mask_agg = observed_mask.sum(dim=-1, keepdim=True).bool() # (B, L, 1)
        
        outputs = self.enc_embedding(observed_tp, observed_data, observed_mask, observed_mask_agg) # (B, L, d_model)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # (B, L, d_model)
        
        n_nonmask = observed_mask_agg.sum(dim=1)  # (B, 1)
        outputs = (outputs * observed_mask_agg).sum(dim=1) / (n_nonmask + 1e-8) # (B, d_model)

        # prediction
        B, Lp = time_steps_to_predict.size()
        time_pred = time_steps_to_predict.unsqueeze(dim = -1) # (B, L, 1)
        h = outputs.unsqueeze(dim=1).repeat(1, Lp, 1) # (B, L, d_model)
        h = torch.cat([h, time_pred], dim=-1) # [B,L,d_model+1]
        output = self.predict_decoder(h).unsqueeze(dim=0) # [1, B, L, D]
        return output #(1, B, L, D)

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
        """ 
        combined_tt: (B, L)
        combined_vals: (B, L, D) tensor containing the observed values.
        combined_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        
        observed_triple, observed_mask = Dataset_MM.collate_fn_triple(observed_tp, observed_data, observed_mask)

        observed_mask = observed_mask.unsqueeze(dim=-1)
        outputs = self.enc_embedding(observed_triple, observed_mask)
        
        outputs = outputs[:,:self.plm_maxlen]
        observed_mask = observed_mask[:,:self.plm_maxlen]

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # (B, n_patch, d_model)
        
        n_nonmask = observed_mask.sum(dim=1)  # (B, 1)
        outputs = (outputs * observed_mask).sum(dim=1) / (n_nonmask + 1e-8) # (B, d_model)

        # prediction
        B, Lp = time_steps_to_predict.size()
        time_pred = time_steps_to_predict.unsqueeze(dim = -1) # (B, L, 1)
        h = outputs.unsqueeze(dim=1).repeat(1, Lp, 1) # (B, L, d_model)
        h = torch.cat([h, time_pred], dim=-1) # [B,L,d_model+1]
        output = self.predict_decoder(h).unsqueeze(dim=0) # [1, B, L, D]
        return output #(1, B, L, D)

class Classifier(nn.Module):

    def __init__(self, dim, cls_dim, activate=None):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(dim, cls_dim)

    def forward(self, ENCoutput):
        ENCoutput = self.linear(ENCoutput)
        return ENCoutput
