import torch as pt

# tensor: pt.Tensor = pt.load('./models/lsknet/lsk_logit_-0.057645753026008606.pt')

def stats(tensor):
    print('Shape: ', tensor.shape)
    print('Max: ', tensor.max())
    print('Min: ', tensor.min())
    # print('Mean: ', tensor.mean())
    print("Any NaN: ", pt.isnan(tensor).any())
    print("Any INF: " ,pt.isinf(tensor).any())
    print("Any -INF: ", pt.isneginf(tensor).any())

# print(tensor)

weights: pt.Tensor = pt.load('./models/lsknet/class_weights_-0.330265074968338.pt')
logits = pt.load('./models/lsknet/logit_-0.330265074968338.pt')
masks = pt.load('./models/lsknet/masks_-0.330265074968338.pt')

stats(weights)
stats(logits)
stats(masks)

loss = pt.nn.functional.cross_entropy(logits, masks, weight=weights)

print(loss)