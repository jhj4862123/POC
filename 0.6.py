import os
import logging
import datetime
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import general_header as gh
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from IPython.display import set_matplotlib_formats
from tqdm import tqdm

set_matplotlib_formats('retina')
now = datetime.datetime.now()
runned = now.strftime("%Y-%m-%d %H:%M:%S")
print(runned + " : 구동")

result_path = '/data/source/POC/plot/' + runned + '/'
os.mkdir(result_path)

#sys.stdout = open(result_path + 'console.txt', 'w')

#photo_path = '/data/workspace/POCdata/photo/ENG'  # 진짜
photo_path = '/data/workspace/POCdata/photo/test' # 테스트
#photo_path = '/data/workspace/POCdata/photo/1/ENG' #학명만
photo_dir = os.listdir(photo_path)

'''########## 로그 파일 생성 ##########'''

dev = [4, 5, 6, 7]
gh.set_gpu(dev)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():

    '''#################################################################### 포맷 및 이미지 크기 설정 #####################################################################'''

    labels = photo_dir
    img_size = 224
    def get_data(data_dir):
        print("numpy에 넣습니다.")
        data = []
        for label in tqdm(labels, desc='get_data'):
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    bgr_arr = cv2.split(img_arr)
                    rgb_arr = cv2.merge([bgr_arr[2], bgr_arr[1], bgr_arr[0]])  # BGR -> RGB 코드
                    resized_arr = cv2.resize(rgb_arr, (img_size, img_size))  # 이미지 크기 변경하기(RGB로)
                    #resized_arr = cv2.resize(img_arr, (img_size, img_size))  # 이미지 크기 변경하기
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)


        return np.array(data, dtype="object")

    #data = get_data('/data/workspace/POCdata/photo/ENG') # 진짜
    #data = get_data('/data/workspace/POCdata/photo/test') # 테스트
    data = get_data('/data/workspace/POCdata/photo/1/ENG')  # 학명만
    print("다 넣음")
    #data = data.shuffle(buffer_size=len(data)) # 시간 잡아먹어서 끔
    '''#################################################################### 총 입력 이미지 수 시각화 #####################################################################'''

    l = []
    for i in data:
        l.append(labels[i[1]])
    sns.set(rc={'figure.figsize': (76.8, 43.2)})
    sns.set_style('dark')
    sns.countplot(l)

    plt.xticks(rotation=90, fontsize=25)
    plt.yticks(rotation=90, fontsize=90)

    plt.savefig(result_path + 'totalamount.png')
    print("TotalImage 출력...")



    '''#################################################################### 입력 이미지 10개 예시 #####################################################################'''

    # total = sns.total_images(data)
    # b = sns.boxplot(x=total["total images"])
    # b.axes.set_title('Total Images')
    # b.set_xlabel('Species', fontsize=15)
    # b.set_ylabel('Amount', fontsize=15)
    # b.tick_params(labelsize=3)
    #
    # fig, ax = plt.subplots(5, 2)Q
    # fig.set_size_inches(15, 15)
    # for i in range(5):
    #     for j in range(2):
    #         l = random.randint(0, len(data))
    #         ax[i, j].imshow(data[l][0])
    #         ax[i, j].set_title('Flower: ' + labels[data[l][1]])
    #
    # plt.tight_layout()
    # plt.savefigresult_path + 'tenflowers.png'

    x = []
    y = []

    print("x,y 배열 삽입")
    for feature, label in tqdm(data):
        x.append(feature)
        y.append(label)
    '''#################################################################### 그레이스케일 정규화 및 리쉐이핑 #####################################################################'''

    x = np.array(x) / 255
    x = x.reshape(-1, img_size, img_size, 3) # Moreover the CNN converges faster on [0..1] data than on [0..255].
    y = np.array(y) # 여기 부분들 오래걸림

    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

    del x, y, data
    print("그레이스케일 완료")

    '''#################################################################### 모델 정의 #####################################################################'''

    print("WORKERS: ", strategy.num_replicas_in_sync)

    pre_trained_model = VGG19(input_shape=(224, 224, 3), include_top=False,  weights="imagenet")  # vgg19를 사용한 이유는 imagenet으로 pre-trained된 모델이기 때문에

    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False

        model = Sequential([
            pre_trained_model,
            MaxPool2D((2, 2), strides=2),
            Flatten(),
            Dense(5, activation='softmax')])
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("Model Summary 완료")
    '''#################################################################### 에포크 #####################################################################'''

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)  # 콜백함수
    history = model.fit(x_train, y_train, batch_size=64, epochs=6, validation_data=(x_test, y_test),callbacks=[learning_rate_reduction])

    print("테스트에 사용할 이미지 수 :", len(y_test))
    print("모델의 Loss는 : - ", model.evaluate(x_test, y_test)[0])
    print("모델의 Accuracy는 : - ", model.evaluate(x_test, y_test)[1] * 100, "%")

    '''#################################################################### 정확도  플롯 #####################################################################'''
    epochs = [i for i in range(6)]
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
    ax[1].set_title('Training & Validation Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.savefig(result_path + 'accuracy & loss.png')

    predictions = model.predict_classes(x_test)
    y_test_inv = label_binarizer.inverse_transform(y_test)

    print(classification_report(y_test_inv, predictions, target_names=labels))
    print("정확도 : ", accuracy_score(y_test_inv, predictions))
    print(predictions)
    print(y_test_inv)



    cm = confusion_matrix(y_test_inv, predictions)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(100, 100))
    sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=labels,
                yticklabels=labels)
    plt.savefig(result_path + 'usedimageamount.png')
    print("UsedImageAmount 출력")

    '''#################################################################### 예측한 결과값 저장 #####################################################################'''

    i = 0
    prop_class = []
    mis_class = []

    for i in range(len(y_test_inv)):
        if (y_test_inv[i] == predictions[i]):
            prop_class.append(i)
        if (len(prop_class) == 8):
            break

    '''#################################################################### 예측과 실제가 일치하는 경우 #####################################################################'''

    i = 0
    for i in range(len(y_test_inv)):
        if (y_test_inv[i] != predictions[i]):
            mis_class.append(i)
        if (len(mis_class) == 8):
            break

    count = 0
    fig, ax = plt.subplots(4, 2)
    fig.set_size_inches(15, 15)
    for i in range(4):
        for j in range(2):
            ax[i, j].imshow(x_test[prop_class[count]])
            ax[i, j].set_title(
                "Predicted Flower : " + labels[predictions[prop_class[count]]] + "\n" + "Actual Flower : " + labels[
                    y_test_inv[prop_class[count]]])
            plt.tight_layout()
            count += 1
    plt.savefig(result_path + 'correctflowers.png')
    print("CorrectFlower 출력" )
    '''#################################################################### 예측과 실제가 일치하지 않는 경우 #####################################################################'''

    count = 0
    fig, ax = plt.subplots(4, 2)
    fig.set_size_inches(15, 15)
    for i in range(4):
        for j in range(2):
            ax[i, j].imshow(x_test[mis_class[count]])
            ax[i, j].set_title(
                "Predicted Flower : " + labels[predictions[mis_class[count]]] + "\n" + "Actual Flower : " + labels[
                    y_test_inv[mis_class[count]]])
            plt.tight_layout()
            count += 1
    plt.savefig(result_path + 'wrongflowers.png')
    print("wrongFlower 출력")

    sys.stdout.close()
    #logging.basicConfig(filename=result_path + '.log', level=logging.DEBUG)
    sys.exit(0)



