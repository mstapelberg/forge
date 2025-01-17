import torch
checkpoint = torch.load('gen_5_model_0-11-28_stagetwo.model', map_location='cpu')
if isinstance(checkpoint, dict):
    print("Keys in checkpoint:", checkpoint.keys())
else:
    print("Direct model save")
