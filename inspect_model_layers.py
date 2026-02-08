import tensorflow as tf
import os
p = os.path.join('backend','model.h5') if os.path.exists(os.path.join('backend','model.h5')) else 'model.h5'
print('Loading', p)
model = tf.keras.models.load_model(p)
for i,l in enumerate(model.layers):
    cname = l.__class__.__name__
    print(i, l.name, cname, 'output_shape=', getattr(l,'output_shape',None))
    cfg = l.get_config()
    # print common config keys
    if 'activation' in cfg:
        print('   activation=', cfg['activation'])
    if 'name' in cfg:
        print('   name=', cfg['name'])
    # print if layer is a Softmax layer
    if cname.lower().find('softmax')!=-1:
        print('   >>> Softmax layer detected')

# Also check model outputs
print('Model outputs:', model.output)
print('Model output shape:', model.output_shape)
