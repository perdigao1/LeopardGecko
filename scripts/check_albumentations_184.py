import leopardgecko.segmentor2 as lgs2

import numpy as np

data = np.random.randint(0,255,size=(184,184,184)).astype(np.uint8)
labels = np.random.randint(0,3, size=(184,184,184))

# augm = lgs2.get_train_augmentations_v1(184,184)

ds0 = lgs2.NN1_train_input_dataset_along_axes([data],[labels])

print(f"{len(ds0)=}")

for i, ds_i in enumerate(ds0):
    data0, label0 = ds_i
    print(f"{i=} , {data0.shape=}, {label0.shape=}")
    if i>10:
        break