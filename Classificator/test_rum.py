from tabnanny import verbose
from matplotlib import pyplot as plt
from numpy import dtype
import tensorflow as tf
from glob import glob
import random
import os 
import numpy as np
import tensorflow.keras as keras
import tensorflow_addons as tfa
import shutil

FILTER_NAME = 'rummi.h5'
FOLDERS = [r'../images/tiles_val']
OUTPATH = r'../images/tiles_err'

all_objects = []
def load_categories(path):
    global all_objects

    #cats = [x[0] for x in os.walk(path)][1:]
    cats = os.listdir(path)
    print(cats)
    print(cats)
    print(cats)
    for c1 in cats:
        c = path +'/'+c1
        if not os.path.isdir(c):
            continue

        cn = c.split('/')[-1].lower()
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

        objs = glob(c+"/*.*")
        for o in objs:
            ext = o.split('.')[-1].lower()
            if ext != 'png' and ext != 'jpg' and ext != 'jpeg':
                continue

            all_objects.append((o,sp[0],sp[1]))

for f in FOLDERS:
    load_categories(f)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 1024
def create_dataset(data):
    def generator():
        for ptr in data:
            print(ptr)
            print(ptr)
            print(ptr)
            print(ptr)
            print(ptr)
            onehot1 = np.zeros(14*4)
            onehot1[ptr[1]*4+ptr[2]] = 1
            yield ptr[0], onehot1


    def decode_img(arg,label1):
        data = tf.io.read_file(arg)
        img = tf.io.decode_jpeg(data, channels=3)
        img = tf.cast(img, tf.float32)/127.5-1
        img = tf.image.resize(img,(32,32), antialias=True)
        return img,label1
    ds = tf.data.Dataset.from_generator(generator, output_types=(tf.string,tf.float32))
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.map(decode_img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds


for i in create_dataset(FOLDERS):
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

import onnxruntime as rt

model = tf.keras.models.load_model(FILTER_NAME)
# providers = ['CPUExecutionProvider']
# m = rt.InferenceSession("../rummi.onnx", providers=providers)

tf.saved_model.save(model, "tmp_model")

model.evaluate(create_dataset(all_objects))
pr = model.predict(create_dataset(all_objects))


colors = ['red','orange','blue','black']
for i, (ptr, batch),  in enumerate(zip(all_objects, create_dataset(all_objects))):
    j = np.argmax(pr[i])
    # onnx_pred = m.run(["tf.reshape_2"], {"input_1": batch[0]})[0]
    # j_onnx = np.argmax(onnx_pred)
    # assert j == j_onnx, i
    cls = j//4
    col = j%4
    path = OUTPATH + '/' + (str(cls)+'_'+colors[col])
    if cls != ptr[1] or col != ptr[2]:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(ptr[0],path+'/'+ptr[0].split('/')[-1])

