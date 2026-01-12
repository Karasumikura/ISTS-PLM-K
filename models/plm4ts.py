import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.Dataset_MM as Dataset_MM

from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope
from transformers.models.bert.modeling_bert_wope import BertModel_wope
from models.embed import *

class ResDecoder(nn.Module):
    """
    Enhanced Decoder with Residual Connections and LayerNorm
    Structure: Linear -> GELU -> [Residual Block: Linear -> GELU -> Dropout -> Linear] -> LayerNorm -> Linear
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ResDecoder, self).__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.act(x)
        
        # Residual Block
        residual = x
        out = self.linear1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        x = self.layer_norm(out + residual) # Add & Norm
        
        x = self.head(x)
        return x

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

        # [MODIFIED] Update decoder to take d_model input (time info is now in embedding)
        # Using a deeper MLP for the generation head
        self.predict_decoder = nn.Sequential(
            nn.Linear(opt.d_model, opt.d_model),
            nn.GELU(),
            nn.Linear(opt.d_model, opt.d_model // 2),
            nn.GELU(),
            nn.Linear(opt.d_model // 2, 1)
        ).to(opt.device)
    
    def step_embedding(self, time_val, value_val):
        """
        Helper function to embed a single autoregressive step.
        time_val: (B*D, 1)
        value_val: (B*D, 1)
        Returns: (B*D, 1, d_model)
        """
        # We manually apply the logic from DataEmbedding_ITS_Ind_VarPrompt
        # 1. Time Embedding
        time_emb = self.enc_embedding.time_embedding(time_val.unsqueeze(-1)) # (B*D, 1, 1, d_model)
        
        # 2. Value Embedding
        # Construct (value, mask) pair. For prediction, we assume mask=1 (valid inference)
        mask_val = torch.ones_like(value_val)
        x_int = torch.cat([value_val.unsqueeze(-1), mask_val.unsqueeze(-1)], dim=-1) # (B*D, 1, 1, 2)
        value_emb = self.enc_embedding.value_embedding(x_int) # (B*D, 1, 1, d_model)
        
        # 3. Combine (Use TE logic)
        if self.enc_embedding.use_te:
            # mask * time + value
            x = mask_val.unsqueeze(-1).unsqueeze(-1) * time_emb + value_emb
        else:
            x = value_emb
            
        return x.squeeze(2) # (B*D, 1, d_model)

    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        """ 
        Autoregressive Forecasting using GPT (gpts[0])
        
        time_steps_to_predict: (B, Lp)
        observed_tp: (B, L, D)
        observed_data: (B, L, D)
        observed_mask: (B, L, D)
        """
        B, L, D = observed_data.shape
        Lp = time_steps_to_predict.size(1)

        # --- 1. Encode History ---
        # Get history embeddings
        # outputs shape: (B*D, L+1, d_model)
        outputs, var_embedding = self.enc_embedding(observed_tp, observed_data, observed_mask) 
        
        # Pass through Temporal GPT (gpts[0])
        # We use use_cache=True to get past_key_values for AR generation
        gpt_out = self.gpts[0](inputs_embeds=outputs, use_cache=True)
        past_key_values = gpt_out.past_key_values
        
        # The last hidden state represents the context up to t_now
        last_hidden = gpt_out.last_hidden_state[:, -1:, :] # (B*D, 1, d_model)

        # --- 2. Autoregressive Generation Loop ---
        predictions = []
        
        # Prepare inputs for the loop
        # Expand time steps for all variables: (B, Lp) -> (B, Lp, D) -> (B*D, Lp)
        time_pred_flat = time_steps_to_predict.unsqueeze(-1).repeat(1, 1, D).reshape(B*D, Lp)
        
        # Initial value for AR: The last observed value in history
        # (B, L, D) -> (B*D, L)
        observed_data_flat = observed_data.permute(0, 2, 1).reshape(B*D, L)
        # Use the last value as the starting point for generation
        current_val = observed_data_flat[:, -1].unsqueeze(-1) # (B*D, 1)

        # If last value was missing (mask=0), we might want to use the model's reconstruction
        # But for simplicity, we start with the data value (0 if missing).
        
        for i in range(Lp):
            # Get target time for this step
            current_time = time_pred_flat[:, i].unsqueeze(-1) # (B*D, 1)
            
            # Embed the current step (Last Value + Next Time)
            step_emb = self.step_embedding(current_time, current_val) # (B*D, 1, d_model)
            
            # GPT Forward (One step)
            gpt_out = self.gpts[0](
                inputs_embeds=step_emb, 
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Update KV cache
            past_key_values = gpt_out.past_key_values
            hidden_state = gpt_out.last_hidden_state # (B*D, 1, d_model)
            
            # Project to scalar value
            # (B*D, 1, d_model) -> (B*D, 1, 1) -> (B*D, 1)
            pred_val = self.predict_decoder(hidden_state).squeeze(-1)
            
            predictions.append(pred_val)
            
            # Update current_val for next iteration (Autoregressive feeding)
            current_val = pred_val

        # --- 3. Format Output ---
        # Stack predictions along time dimension
        results = torch.cat(predictions, dim=1) # (B*D, Lp)
        
        # Reshape back to (1, B, Lp, D) as expected by the framework
        # (B*D, Lp) -> (B, D, Lp) -> (B, Lp, D)
        results = results.reshape(B, D, Lp).permute(0, 2, 1)
        
        return results.unsqueeze(0) # (1, B, Lp, D)
    
        
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
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
        
        ### avg pooling
        n_nonmask = observed_mask.sum(dim=1)  # (B*D, 1)
        outputs = (outputs * observed_mask).sum(dim=1) / n_nonmask # (B*D, d_model)
        outputs = self.ln_proj(outputs.view(B, D, -1))  # (B, D, d_model)

        outputs = outputs + var_embedding.squeeze()
        outputs = self.gpts[1](inputs_embeds=outputs).last_hidden_state # (B, D, d_model)

        # forcasting
        B, Lp = time_steps_to_predict.size()
        time_pred = time_steps_to_predict.unsqueeze(dim = -1).repeat(1, 1, D).unsqueeze(dim = -1) # (B, L, D, 1)
        h = outputs.unsqueeze(dim=1).repeat(1, Lp, 1, 1) # (B, L, D, d_model)
        h = torch.cat([h, time_pred], dim=-1) # [B,L,D,d_model+1]
        output = self.predict_decoder(h).unsqueeze(dim=0) # [1, B,L,D, 1]
        output = output.squeeze(dim = -1)
        return output #(1, B, L, D)

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
                
        # Enhanced Decoder
        self.predict_decoder = ResDecoder(
            input_dim=opt.d_model+1, 
            hidden_dim=opt.d_model, 
            output_dim=opt.input_dim,
            dropout=opt.dropout if hasattr(opt, 'dropout') else 0.1
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

        # Enhanced Decoder
        self.predict_decoder = ResDecoder(
            input_dim=opt.d_model+1, 
            hidden_dim=opt.d_model, 
            output_dim=opt.input_dim,
            dropout=opt.dropout if hasattr(opt, 'dropout') else 0.1
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
