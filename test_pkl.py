import torch as th

model = th.load("/mnt/data1/wx/head_angle_correct/model_save/unet_modify_group.pkl")

print(model.keys())
