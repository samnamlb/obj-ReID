import torch
from models.model import StudentModel

model = StudentModel(device="cpu")

dummy = torch.randn(4, 3, 224, 224)
emb = model.encode(dummy)

print("Shape:", emb.shape)
print("Norms:", emb.norm(dim=1))

img = torch.randn(1, 3, 224, 224)

e1 = model.encode(img)
e2 = model.encode(img)

print(torch.allclose(e1, e2))