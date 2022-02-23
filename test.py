import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import general_header as gh
import tensorflow as tf
import tensorflow.keras
import numpy as np
from tensorflow.keras.models import load_model

dev = [7]
gh.set_gpu(dev)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    photo_path = '/data/workspace/POCdata/photo/poc'  # poc
    labels = os.listdir(photo_path)
    test_path = '/data/workspace/POCdata/photo/test/'  # test

    model = load_model('1.1sigmodel.h5')

    img_size = 224

    a = sys.argv[1]
    img_path = '/data/workspace/POCdata/photo/test/' + a + '.jpg'
    img = tensorflow.keras.preprocessing.image.load_img(
        img_path, target_size=(img_size, img_size)
    )
    img_array = tensorflow.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = np.array(img_array) / 255  # 데이터 노멀라이징
    img_array = img_array.reshape(-1, img_size, img_size, 3)  # 3차원 배열화


    predictions = model.predict(img_array)
    #softscore = tf.nn.softmax(predictions)

    # print(
    #          "이 이미지는 {}라고 분류됩니다."
    #              .format(labels[np.argmax(predictions)]))

    if 100 * np.max(predictions) < 15:
        print("이 이미지는 확률이 너무 낮아 분류할 수 없습니다.")

    else:
        print(
            "입력한 이미지 " + str(a) + " (은)는 {}라고 분류됩니다. 정확도는 {:.2f}%입니다."
                .format(labels[np.argmax(predictions)], 100 * np.max(predictions)))
