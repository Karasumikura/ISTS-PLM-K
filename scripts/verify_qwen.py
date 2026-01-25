import torch
import torch.nn as nn
from models.plm4ts import istsplm_forecast

class Opts:
    def __init__(self):
        self.input_dim = 1
        self.d_model = 896 # Qwen2.5-0.5B default
        self.input_len = 96
        self.device = 'cpu'
        self.dropout = 0.1
        self.te_model = 'qwen'
        self.st_model = 'gpt' # Keep 2nd model as GPT for now to test mixed usage
        self.n_te_plmlayer = 1
        self.n_st_plmlayer = 1
        self.semi_freeze = False
        self.max_len = 512

def test_qwen_integration():
    opt = Opts()
    print("Initializing model...")
    try:
        model = istsplm_forecast(opt)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    print("Checking if Gpts[0] is Qwen...")
    print(type(model.gpts[0]))

    # Dummy input
    B, L, D = 2, 96, 1
    observed_tp = torch.rand(B, L, D).to(opt.device)
    observed_data = torch.rand(B, L, D).to(opt.device)
    observed_mask = torch.ones(B, L, D).to(opt.device)
    time_steps_to_predict = torch.rand(B, 24).to(opt.device) 

    # Mocking PLMs folder existence for testing if not present?
    # No, we assume user hasn't downloaded it yet, so this might fail if model not found.
    # But we want to check if the CODE path works. 
    # If model not found, it should raise OSError.
    
    try:
        print("Running forward pass...")
        output = model.forecasting(time_steps_to_predict, observed_data, observed_tp, observed_mask)
        print("Forward pass successful.")
        print("Output shape:", output.shape)
    except OSError as e:
        print("Model files not found (Expected). Please ensure ./PLMs/qwen2.5-0.5b exists.")
        print(e)
    except Exception as e:
        print(f"Runtime error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen_integration()
