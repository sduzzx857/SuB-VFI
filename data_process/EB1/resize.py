import tifffile as tf           #tifffile是tiff文件的读取库
import shutil
import os

def resize(h, w):
    root = 'data_process/EB1/EB1-origin'
    save_train = 'data_process/EB1/train'
    save_test = 'data_process/EB1/test'
    os.makedirs(save_test, exist_ok=True)
    os.makedirs(save_train, exist_ok=True)
    testdir = sorted(os.listdir(root))
    for i, d in enumerate(testdir[28:]):
        tif_path = os.path.join(root, d)
        img_tf = tf.imread(tif_path)
        ih, iw, _ = img_tf.shape
        y = int((ih-h) )
        x = int((iw-w) / 2)
        if i < 34:
            save_path = os.path.join(save_train, d)
        else:
            save_path = os.path.join(save_test, d)
        tf.imwrite(save_path, img_tf[y:y+h, x:x+w, 0])
    
    shutil.rmtree(root)

resize(352,288)