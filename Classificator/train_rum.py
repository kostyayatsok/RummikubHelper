from tabnanny import verbose
from numpy import dtype
import tensorflow as tf
from glob import glob
import random
import os 
import numpy as np
import tensorflow.keras as keras
import tensorflow_addons as tfa


FILTER_NAME = 'rummi.h5'
FOLDERS = [r'D:\databases\tiles_aligned']

def add_l2_regularizer_2_model(model, weight_decay, custom_objects={}, apply_to_batch_normal=False, apply_to_bias=False):
    # https://github.com/keras-team/keras/issues/2717#issuecomment-456254176
    for layer in model.layers:
        attrs = []
        if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D):
            attrs = ["kernel_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            # print(">>>> DepthwiseConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.SeparableConv2D):
            # print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["pointwise_regularizer", "depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.BatchNormalization):
            # print(">>>> BatchNormalization", layer.name, "scale:", layer.scale, ", center:", layer.center)
            if layer.center:
                attrs.append("beta_regularizer")
            if layer.scale:
                attrs.append("gamma_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.PReLU):
            # print(">>>> PReLU", layer.name)
            attrs = ["alpha_regularizer"]

        for attr in attrs:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, keras.regularizers.L2(weight_decay / 2))

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    # temp_weight_file = "tmp_weights.h5"
    # model.save_weights(temp_weight_file)
    # out_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)
    # out_model.load_weights(temp_weight_file, by_name=True)
    # os.remove(temp_weight_file)
    # return out_model
    return keras.models.clone_model(model)

all_objects = []
def load_categories(path):
    global all_categories
    global all_objects

    #cats = [x[0] for x in os.walk(path)][1:]
    cats = os.listdir(path)
    for c1 in cats:
        c = path +'\\'+c1
        if not os.path.isdir(c):
            continue

        cn = c.split('\\')[-1].lower()
        if cn == 'class':
            continue
        if cn == '_exclude':
            continue

        sp = cn.split('-')
        if len(sp) != 2:
            print("Unable parse", cn)
            continue

        if sp[0] == 'j':
            sp[0] = '0'

        sp[0] = int(sp[0])
        if sp[1] == 'red':
            sp[1] = 0
        elif sp[1] == 'orange':
            sp[1] = 1
        elif sp[1] == 'blue':
            sp[1] = 2
        elif sp[1] == 'black':
            sp[1] = 3
        else:
            print("Unable parse", cn)
            continue

        objs = glob(c+"\\*.*")
        for o in objs:
            ext = o.split('.')[-1].lower()
            if ext != 'png' and ext != 'jpg' and ext != 'jpeg':
                continue

            all_objects.append((o,sp[0],sp[1]))

load_categories(r'D:\databases\tiles_aligned')
train    = all_objects
all_objects = []
load_categories(r'D:\databases\tiles_val')
validate = all_objects

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 512
img_sz = 32
#trainlen = len(train)//batch_size
#vallen = len(validate)//batch_size
def create_dataset(data, rotate):

    def generator():
        for ptr in data:
            onehot1 = np.zeros(14*4)
            onehot1[ptr[1]*4+ptr[2]] = 1
            yield ptr[0], onehot1
    def decode_img(arg,label1):
        data = tf.io.read_file(arg)
        img = tf.io.decode_jpeg(data, channels=3)
        img = tf.cast(img, tf.float32)/127.5
        img = tf.image.resize(img,(2*img_sz,2*img_sz),antialias=True)
        return img,label1
    def random_rotate(img,label1):
        img = tfa.image.rotate(img,tf.random.uniform((1,),-180,180),interpolation='bilinear')
        size = tf.random.uniform((2,),img_sz//2,img_sz*3//2,dtype=tf.int32)
        img = tf.image.resize(img,size,antialias=True)
        img = tf.image.pad_to_bounding_box(img,(img_sz-tf.minimum(size[0],img_sz))//2,(img_sz-tf.minimum(size[1],img_sz))//2,tf.maximum(size[0],img_sz),tf.maximum(size[1],img_sz))
        img = tf.image.random_crop(img,(img_sz,img_sz,3))
        imgbkg = (img[:,:] == tf.zeros((3,)))
        imgbkg = tf.cast(imgbkg,tf.float32)
        img = img*(1 - imgbkg) + imgbkg*tf.random.uniform((1,3),0.,2.)
        img = tf.image.random_brightness(img,1.0)
        img = tf.image.random_contrast(img,0.2,2.)
        return tf.clip_by_value(img-1,-1,1),label1
    def resize(img,label1):
        img = tf.image.resize(img,(img_sz,img_sz),antialias=True)
        return tf.clip_by_value(img-1,-1,1),label1
    ds = tf.data.Dataset.from_generator(generator, output_types=(tf.string,tf.float32))
    ds = ds.map(decode_img, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if rotate:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(resize, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.shuffle(1024*16)
    ds = ds.batch(batch_size)
    return ds

train = create_dataset(train, True)
validate = create_dataset(validate, False)


import matplotlib.pyplot as plt
for i in validate:
    showed = 0
    fig = plt.figure(figsize=(10,10))
    for img in i[0]:
        plt.subplot(4,8,showed+1) # <-------
        plt.imshow((img+1)/2)
        showed = showed + 1
        if showed == 31:
            break
    plt.show()
    break

for i in train:
    showed = 0
    fig = plt.figure(figsize=(10,10))
    for img in i[0]:
        plt.subplot(4,8,showed+1) # <-------
        plt.imshow((img+1)/2)
        showed = showed + 1
        if showed == 31:
            break
    plt.show()
    break
# img112 =tf.keras.layers.Input((32,32,3))
img112 =tf.keras.layers.Input((img_sz,img_sz,3))
# classifier = tf.keras.applications.MobileNetV2(include_top=False,input_shape=(32,32,3),weights='imagenet',classes = None)
#classifier = tf.keras.applications.MobileNetV2(include_top=False,input_shape=(img_sz,img_sz,3),weights='imagenet',classes = None)
from vgg8 import vgg8
classifier = vgg8((img_sz,img_sz,3))
res = classifier(img112)
res = tf.keras.layers.Flatten()(res)
#res = tf.keras.layers.Dropout(0.4)(res)
res = tf.keras.layers.Dense(128, activation = 'relu')(res)
res = tf.keras.layers.BatchNormalization()(res)
res = tf.linalg.l2_normalize(res,axis=-1)
res1 = tf.keras.layers.Dense(14, activation = 'softmax', use_bias = False)(res)
res2 = tf.keras.layers.Dense(4, activation = 'softmax', use_bias = False)(res)
res = tf.reshape(res1,(-1,14,1))*tf.reshape(res2,(-1,1,4))
res = tf.reshape(res,(-1,14*4))
model = tf.keras.models.Model(img112,res)
try:
    model = tf.keras.models.load_model(FILTER_NAME+1)
except:
    print("Weights not loaded")

model = add_l2_regularizer_2_model(model, 0.01)
#model.save(FILTER_NAME)

def set_batchnorm_momentum(model, m):
    def _set_batchnorm_momentum(layer, m):
        if isinstance(layer, tf.keras.Model):
            for l in layer.layers:
                _set_batchnorm_momentum(l, m)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = m
    _set_batchnorm_momentum(model, m)
    return

set_batchnorm_momentum(model,0.95)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum = 0.9)
#optimizer = tfa.optimizers.SGDW(learning_rate=1e-2, momentum=0.9, weight_decay=1e-3)
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=FILTER_NAME,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True,
    verbose=True)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=15,
    verbose=1,
)

model.fit(train, validation_data = validate, epochs = 1024,
    callbacks=
    [model_checkpoint_callback,
    reduce_lr_callback
    ])

