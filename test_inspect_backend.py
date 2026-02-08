import backend.app as b

print('mdl is None?', b.mdl is None)
print('use_sigmoid_fix:', b.use_sigmoid_fix)
print('penultimate_model is None?:', b.penultimate_model is None)
try:
    if b.final_W is not None:
        import numpy as np
        print('final_W shape:', np.array(b.final_W).shape)
        print('final_b shape:', np.array(b.final_b).shape)
except Exception as e:
    print('inspect error:', e)
