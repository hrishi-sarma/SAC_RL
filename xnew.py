# import torch
# print(torch.cuda.is_available())       # True
# print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 4060 ...

import os
print(os.cpu_count())  # use this number for set_num_threads