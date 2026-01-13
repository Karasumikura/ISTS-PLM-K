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
        # Disable additive TimeEmbedding (APE) by passing use_te=False
        self.enc_embedding = DataEmbedding_ITS_Ind_VarPrompt(self.feat_dim, self.d_model, self.feat_dim, device=opt.device, dropout=opt.dropout, use_te=False)

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
            elif(i==2): # Decoder (Sequence Generator)
                # Use 'te_model' architecture for decoder as it processes time sequence
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

        # [MODIFIED] Decoder components instead of MLP
        # 1. Time Embedding to project scalar time to d_model
        self.decoder_time_emb = TimeEmbedding(self.d_model).to(opt.device)
        # 2. Final Projection
        self.decoder_out_proj = nn.Linear(self.d_model, 1).to(opt.device)
    
        
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        """ 
        observed_tp: (B, L, D)
        observed_data: (B, L, D) tensor containing the observed values.
        observed_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
        """
    
        B, L, D = observed_data.shape
        
        # --- PART 1: ENCODING HISTORY ---
        outputs, var_embedding = self.enc_embedding(observed_tp, observed_data, observed_mask) # (B*D, L+1, d_model)
        
        # Prepare time_ids for RoPE
        # observed_tp: (B, L, D) -> permute to (B, D, L) -> reshape to (B*D, L)
        time_ids = observed_tp.permute(0, 2, 1).reshape(B*D, L)
        
        # The prompt token (first position) needs a timestamp. We'll use the first time step's value
        # or 0 as a dummy timestamp. Let's use the first timestep for each series.
        # time_ids: (B*D, L) -> we need (B*D, L+1) to include the prompt
        # Use the first time value as the prompt's timestamp
        prompt_time = time_ids[:, 0:1]  # (B*D, 1)
        time_ids = torch.cat([prompt_time, time_ids], dim=1)  # (B*D, L+1)
        
        # Pass time_ids to gpts[0] for time-based RoPE
        outputs = self.gpts[0](inputs_embeds=outputs, time_ids=time_ids).last_hidden_state # (B*D, L+1, d_model)

        observed_mask = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1) # (B*D, L, 1)
        observed_mask = torch.cat([torch.ones_like(observed_mask[:,:1]), observed_mask], dim=1) # (B*D, L+1, 1)
        
        ### avg pooling
        n_nonmask = observed_mask.sum(dim=1)  # (B*D, 1)
        outputs = (outputs * observed_mask).sum(dim=1) / n_nonmask # (B*D, d_model)
        outputs = self.ln_proj(outputs.view(B, D, -1))  # (B, D, d_model)

        outputs = outputs + var_embedding.squeeze()
        outputs = self.gpts[1](inputs_embeds=outputs).last_hidden_state # (B, D, d_model)
        
        # outputs now contains the "Context Vector" for each variable: (B, D, d_model)

        # --- PART 2: DECODING FUTURE (Using 3rd PLM) ---
        B, Lp = time_steps_to_predict.size()
        
        # 1. Embed Future Time Steps
        # time_steps_to_predict: (B, Lp) -> (B, Lp, 1)
        time_pred = time_steps_to_predict.unsqueeze(-1)
        # Map time to d_model using TimeEmbedding
        # Result: (B, Lp, d_model)
        time_emb = self.decoder_time_emb(time_pred) 
        
        # 2. Combine Context + Future Time
        # Context (outputs): (B, D, d_model) -> (B, D, 1, d_model)
        # Future (time_emb): (B, Lp, d_model) -> (B, 1, Lp, d_model)
        # We add them to "condition" the future time steps on the variable's history context
        # Result: (B, D, Lp, d_model)
        decoder_input = outputs.unsqueeze(2) + time_emb.unsqueeze(1)
        
        # 3. Reshape for PLM Processing
        # We treat each variable's forecast as an independent sequence of length Lp
        # Shape: (B * D, Lp, d_model)
        decoder_input = decoder_input.view(B*D, Lp, self.d_model)
        
        # 4. Pass through 3rd PLM (Decoder)
        # The PLM models the dependencies between the Lp future points (smoothing, trends)
        dec_out = self.gpts[2](inputs_embeds=decoder_input).last_hidden_state # (B*D, Lp, d_model)
        
        # 5. Project to Value
        pred = self.decoder_out_proj(dec_out) # (B*D, Lp, 1)
        
        # 6. Reshape to required output format (1, B, Lp, D)
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
