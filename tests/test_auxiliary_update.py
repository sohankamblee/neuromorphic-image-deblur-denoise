import numpy as np

from src.optimization.auxiliary_update import update_auxiliary

image = np.random.rand(260,346).astype(np.float32)

zh, zv = update_auxiliary(
    image,
    beta=0.064,
    gamma=2.0,
)

print("zh:", zh.shape)
print("zv:", zv.shape)

print("Non-zero zh:", np.count_nonzero(zh))
print("Non-zero zv:", np.count_nonzero(zv))