import torch
print(torch.cuda.is_available())  # Debería devolver True si CUDA está disponible
print(torch.cuda.current_device())  # El número del dispositivo GPU activo
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # El nombre de la GPU
