import torch
import matplotlib.pyplot as plt
import numpy as np

v  = 26
d_model = 26
pos = 10

torch.manual_seed(42)

W_U = torch.randint(-3, 0, (d_model, v), dtype=torch.float32) 
W_E = torch.randint(0, 4, (v, d_model), dtype=torch.float32) 
X = torch.randint(0, 2, (pos, d_model), dtype=torch.float32)  

W_UE = W_U @ W_E
X_WUE = X @ W_UE

fig, axes = plt.subplots(1, 5, figsize=(10, 10))

matrices = [W_U, W_E, W_UE, X, X_WUE]
titles = ["Wu", "We", "Wue", "X", "X @ Wue"]

for ax, mat, title in zip(axes, matrices, titles):
    im = ax.imshow(mat.numpy(), cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "", ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# import torch
# import matplotlib.pyplot as plt

# W_U = torch.tensor([
#     [ 1.,  0.,  2., -1.],
#     [ 0.,  1.,  1.,  0.],
#     [-1.,  2.,  0.,  1.],
# ])

# W_E = torch.tensor([
#     [ 1.,  0.,  1.],
#     [ 0.,  1., -1.],
#     [ 1.,  1.,  0.],
#     [ 2., -1.,  1.],
# ])

# X = torch.tensor([
#     [1., 0., 1.],
#     [0., 1., 1.],
#     [1., 1., 0.],
# ])

# M = W_U @ W_E
# Y = X @ M

