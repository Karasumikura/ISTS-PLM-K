import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.Dataset_MM as Dataset_MM

from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope
from transformers.models.bert.modeling_bert_wope import BertModel_wope
from models.embed import *

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
        
        # Generate time_ids from observed_tp for RoPE
        # observed_tp: (B, L, D) -> (B*D, L)
        # We need to add one more time step for the prompt token
        observed_tp_reshaped = observed_tp.permute(0, 2, 1).reshape(B*D, L)  # (B*D, L)
        # Add a zero time step at the beginning for the prompt token
        time_ids = torch.cat([torch.zeros_like(observed_tp_reshaped[:, :1]), observed_tp_reshaped], dim=1)  # (B*D, L+1)
        
        outputs = self.gpts[0](inputs_embeds=outputs, time_ids=time_ids).last_hidden_state # (B*D, L+1, d_model)
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

# =================================================================================
# [INNOVATION CORE] Modified Class with 3rd PLM Decoder, Concatenation & Anchoring
# =================================================================================
class istsplm_forecast(nn.Module):
    
    def __init__(self, opt):
        super(istsplm_forecast, self).__init__()
        self.feat_dim = opt.input_dim
        self.d_model = opt.d_model
        self.input_len = opt.input_len
        self.enc_embedding = DataEmbedding_ITS_Ind_VarPrompt(self.feat_dim, self.d_model, self.feat_dim, device=opt.device, dropout=opt.dropout)

        self.gpts = nn.ModuleList()
        # [MODIFIED] Increased range to 3 to include Decoder PLM
        for i in range(3):
            gpt2 = GPT2Model_wope.from_pretrained('./PLMs/gpt2', output_attentions=True, output_hidden_states=True)
            bert = BertModel_wope.from_pretrained('./PLMs/bert-base-uncased', output_attentions=True, output_hidden_states=True)
            print(f"Loading PLM {i}: {opt.te_model if i != 1 else opt.st_model}")
            
            if(i==0): # Time Encoder (Intra-series)
                if opt.te_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_te_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.te_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_te_plmlayer]
                    self.gpts.append(bert)
            elif(i==1): # Space Encoder (Inter-series)
                if opt.st_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_st_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.st_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_st_plmlayer]
                    self.gpts.append(bert)
            elif(i==2): # [NEW] Decoder (Sequence Generator)
                # Using 'te_model' architecture for decoder as it processes time sequence
                if opt.te_model == 'gpt':
                    gpt2.h = gpt2.h[:opt.n_te_plmlayer]
                    self.gpts.append(gpt2)
                elif opt.te_model == 'bert':
                    bert.encoder.layer = bert.encoder.layer[:opt.n_te_plmlayer]
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

        # [MODIFIED] Decoder components (Replacing simple MLP)
        
        # 1. Time Embedding: Projects prediction timestamps to d_model
        self.decoder_time_emb = TimeEmbedding(self.d_model).to(opt.device)
        
        # 2. [NEW] Last Value Projection (Anchor): Projects scalar value to d_model
        self.decoder_val_proj = nn.Linear(1, self.d_model).to(opt.device)

        # 3. [NEW] Fusion Layer: Projects concatenated features (3*d_model) back to d_model
        # Inputs: [Context (d_model), FutureTime (d_model), LastValue (d_model)]
        self.fusion_layer = nn.Linear(3 * self.d_model, self.d_model).to(opt.device)

        # 4. Final Projection: Projects latent state to scalar prediction
        self.decoder_out_proj = nn.Linear(self.d_model, 1).to(opt.device)
    
        
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        """ 
        observed_tp: (B, L, D)
        observed_data: (B, L, D) tensor containing the observed values.
        observed_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        
        B, L, D = observed_data.shape
        
        # --- PART 1: ENCODING HISTORY (Standard ISTS-PLM Logic) ---
        outputs, var_embedding = self.enc_embedding(observed_tp, observed_data, observed_mask) # (B*D, L+1, d_model)
        
        # Generate time_ids from observed_tp for RoPE
        # observed_tp: (B, L, D) -> (B*D, L)
        # We need to add one more time step for the prompt token
        observed_tp_reshaped = observed_tp.permute(0, 2, 1).reshape(B*D, L)  # (B*D, L)
        # Add a zero time step at the beginning for the prompt token
        time_ids = torch.cat([torch.zeros_like(observed_tp_reshaped[:, :1]), observed_tp_reshaped], dim=1)  # (B*D, L+1)
        
        outputs = self.gpts[0](inputs_embeds=outputs, time_ids=time_ids).last_hidden_state # (B*D, L+1, d_model)

        observed_mask = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1) # (B*D, L, 1)
        observed_mask = torch.cat([torch.ones_like(observed_mask[:,:1]), observed_mask], dim=1) # (B*D, L+1, 1)
        
        ### avg pooling to get Context Vector
        n_nonmask = observed_mask.sum(dim=1)  # (B*D, 1)
        outputs = (outputs * observed_mask).sum(dim=1) / n_nonmask # (B*D, d_model)
        outputs = self.ln_proj(outputs.view(B, D, -1))  # (B, D, d_model)

        outputs = outputs + var_embedding.squeeze()
        # Pass through Variable-Aware PLM (Inter-series modeling)
        outputs = self.gpts[1](inputs_embeds=outputs).last_hidden_state # (B, D, d_model)
        
        # 'outputs' is now the Context Vector for each variable

        # --- PART 2: DECODING FUTURE (Modified Logic) ---
        B, Lp = time_steps_to_predict.size()
        
        # 1. Prepare Future Time Embedding
        # (B, Lp) -> (B, Lp, 1) -> (B, Lp, d_model)
        time_pred = time_steps_to_predict.unsqueeze(-1)
        time_emb = self.decoder_time_emb(time_pred) 
        
        # 2. Prepare Last Known Value (Anchor)
        # [NEW] Extract last time step value as anchor
        # Note: For strict ISTS, robust imputation (e.g., forward fill) might be needed here
        last_vals = observed_data[:, -1, :] # (B, D)
        last_vals_emb = self.decoder_val_proj(last_vals.unsqueeze(-1)) # (B, D, 1) -> (B, D, d_model)
        
        # 3. Expand All Features for Concatenation (Align dimensions)
        
        # A. Context: (B, D, d_model) -> (B, D, Lp, d_model)
        # Repeats the static context for every future time step
        outputs_expanded = outputs.unsqueeze(2).expand(-1, -1, Lp, -1)
        
        # B. Future Time: (B, Lp, d_model) -> (B, 1, Lp, d_model) -> (B, D, Lp, d_model)
        # Shared across all variables (assuming synchronized prediction grid)
        time_emb_expanded = time_emb.unsqueeze(1).expand(B, D, -1, -1)
        
        # C. Last Value: (B, D, d_model) -> (B, D, Lp, d_model)
        # Repeats the anchor value for every future time step
        last_vals_expanded = last_vals_emb.unsqueeze(2).expand(-1, -1, Lp, -1)
        
        # 4. Concatenation & Fusion [NEW]
        # Concatenate Context, Time, and Anchor along feature dimension
        # Shape: (B, D, Lp, 3 * d_model)
        decoder_input_cat = torch.cat([outputs_expanded, time_emb_expanded, last_vals_expanded], dim=-1)
        
        # Fuse features: (B, D, Lp, 3*d_model) -> (B, D, Lp, d_model)
        decoder_input = self.fusion_layer(decoder_input_cat)
        
        # 5. Reshape for PLM Processing
        # Flatten B and D to treat each variable series independently in the decoder
        # Shape: (B * D, Lp, d_model)
        decoder_input = decoder_input.view(B*D, Lp, self.d_model)
        
        # 6. Pass through 3rd PLM (Decoder)
        # Models dependencies between future time steps
        dec_out = self.gpts[2](inputs_embeds=decoder_input).last_hidden_state # (B*D, Lp, d_model)
        
        # 7. Project to Value
        pred = self.decoder_out_proj(dec_out) # (B*D, Lp, 1)
        
        # 8. Reshape to required output format (1, B, Lp, D)
        # (B*D, Lp, 1) -> (B, D, Lp) -> (B, Lp, D)
        pred = pred.view(B, D, Lp).permute(0, 2, 1)
        
        return pred.unsqueeze(0) # (1, B, Lp, D)

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
