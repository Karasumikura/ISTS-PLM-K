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
            # print(f"Loading PLM {i}: {opt.te_model if i != 1 else opt.st_model}")
            
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
        
        # Freezing logic omitted for brevity (same as before)...
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

        # --- Decoder Components ---
        # 1. Time Embedding (Predict Time -> d_model)
        self.decoder_time_emb = TimeEmbedding(self.d_model).to(opt.device)
        
        # [NEW] 2. Last Value Embedding (Scalar Value -> d_model)
        # 用于把上一个时刻的真实观测值映射到高维空间，作为Decoder的数值锚点
        self.decoder_val_proj = nn.Linear(1, self.d_model).to(opt.device)

        # [NEW] 3. Fusion Layer (Concatenation -> d_model)
        # 输入维度是 3 * d_model (Context + FutureTime + LastValue)
        self.fusion_layer = nn.Linear(3 * self.d_model, self.d_model).to(opt.device)

        # 4. Final Projection
        self.decoder_out_proj = nn.Linear(self.d_model, 1).to(opt.device)
    
        
    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        """ 
        observed_data: (B, L, D) 
        """
        
        B, L, D = observed_data.shape
        
        # --- PART 1: ENCODING HISTORY (保持原样) ---
        outputs, var_embedding = self.enc_embedding(observed_tp, observed_data, observed_mask) 
        outputs = self.gpts[0](inputs_embeds=outputs).last_hidden_state 

        observed_mask = observed_mask.permute(0, 2, 1).reshape(B*D, -1, 1) 
        observed_mask = torch.cat([torch.ones_like(observed_mask[:,:1]), observed_mask], dim=1) 
        
        n_nonmask = observed_mask.sum(dim=1)  
        outputs = (outputs * observed_mask).sum(dim=1) / n_nonmask 
        outputs = self.ln_proj(outputs.view(B, D, -1))  
        outputs = outputs + var_embedding.squeeze()
        outputs = self.gpts[1](inputs_embeds=outputs).last_hidden_state 
        
        # outputs (Context Vector): (B, D, d_model)

        # --- PART 2: DECODING FUTURE (Modified Logic) ---
        B, Lp = time_steps_to_predict.size()
        
        # 1. Prepare Future Time Embedding
        # (B, Lp) -> (B, Lp, d_model)
        time_pred = time_steps_to_predict.unsqueeze(-1)
        time_emb = self.decoder_time_emb(time_pred) 
        
        # 2. Prepare Last Known Value (Anchor)
        # [NEW] 提取最后一个时间步的值作为锚点
        # 注意：这里简单取了 input 的最后一步。对于 ISTS，如果最后一步缺失建议先做简单填充(Imputation)
        last_vals = observed_data[:, -1, :] # (B, D)
        last_vals_emb = self.decoder_val_proj(last_vals.unsqueeze(-1)) # (B, D, 1) -> (B, D, d_model)
        
        # 3. Expand All Features for Concatenation
        # 我们需要把所有特征对齐到 (B, D, Lp, d_model)
        
        # A. Context: (B, D, d_model) -> (B, D, Lp, d_model)
        outputs_expanded = outputs.unsqueeze(2).expand(-1, -1, Lp, -1)
        
        # B. Future Time: (B, Lp, d_model) -> (B, 1, Lp, d_model) -> (B, D, Lp, d_model)
        time_emb_expanded = time_emb.unsqueeze(1).expand(B, D, -1, -1)
        
        # C. Last Value: (B, D, d_model) -> (B, D, Lp, d_model)
        # 这里我们将“最后已知值”复制到未来的每一步，作为静态参考
        last_vals_expanded = last_vals_emb.unsqueeze(2).expand(-1, -1, Lp, -1)
        
        # 4. Concatenation & Fusion [NEW]
        # 以前是相加 (+)，现在是拼接 (cat)
        # Shape: (B, D, Lp, 3 * d_model)
        decoder_input_cat = torch.cat([outputs_expanded, time_emb_expanded, last_vals_expanded], dim=-1)
        
        # Shape: (B, D, Lp, d_model)
        decoder_input = self.fusion_layer(decoder_input_cat)
        
        # 5. Reshape for PLM Processing
        # Shape: (B * D, Lp, d_model)
        decoder_input = decoder_input.view(B*D, Lp, self.d_model)
        
        # 6. Pass through 3rd PLM (Decoder)
        dec_out = self.gpts[2](inputs_embeds=decoder_input).last_hidden_state 
        
        # 7. Project to Value
        pred = self.decoder_out_proj(dec_out) # (B*D, Lp, 1)
        
        # 8. Reshape to output format
        pred = pred.view(B, D, Lp).permute(0, 2, 1)
        
        return pred.unsqueeze(0) # (1, B, Lp, D)
