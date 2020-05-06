import os.path
import gc
from sklearn.externals import joblib  # jbolib模組
from Load_Metadata import load_metadata,normalization,label,distance,recordToCSV,move,copyimg,load_metadata_predict
from align import AlignDlib
from model import create_model

print("set path--------------------")
#set path
path_database='./images/'
path_recognize='./Taipei/Check'
path_recognize_folder = './Taipei/Check/folder'
path_finish = './Taipei/RecognizePhoto'
path_finish2 ='./Taipei/RecognizePhoto2'


classnum = len(os.listdir(path_database)) #資料庫有幾個人
print("classnum: "  ,classnum)
distance_thresholde=0.5


print("load model------------------")
#load model
svc = joblib.load('models/svc.pkl')
knn = joblib.load('models/knn.pkl')
#set flag
RecognizeFlag = 1

print("prepare database")
# prepare before recognize
# 讀取database 照片
metadata_database = load_metadata(path_database)
# 原始圖片正規化
recognize_flag_metadata_database, embedded_metadata_database,i_metadata_database = \
    normalization(metadata_database)
encoder = label(metadata_database)

print("set nn4 models---------------------")
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('models/nn4.small2.v1.h5')
print("start")


#--------------------------------------------

import base64, json
import numpy as np
from flask import Flask, request
from flask_cors import CORS
import  cv2
app=Flask(__name__)
CORS(app, resources=r'/*')


def normalization_img(img):
    def load_image(img):

        return img[..., ::-1]

    def align_image(img):
        return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                               landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    alignment = AlignDlib('models/landmarks.dat')
    embedded = np.zeros((1, 128))

    try:
        img = load_image(img)
        img = align_image(img)
        img = (img / 255.).astype(np.float32)
        embedded[1] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        recognize_flag = 1

        del img

    except Exception as e:
        recognize_flag = 0
    return recognize_flag, embedded


@app.route('/', methods=['GET', 'POST'])
def root():
    response = []
    # 從 flask request 中讀取圖片（byte str）
    image=request.form['image'].replace(' ', '+')

    # 做 base64 的解碼
    image=base64.b64decode(image)
    img_array = np.fromstring(image, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # 顯示圖片
    cv2.imshow('My Image', img)

    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    recognize_flag, embedded = normalization_img(img)
    print("1 recognize_flag :", recognize_flag)

    if recognize_flag == 0:
        print("擷取人臉特徵失敗")

        img_copy = img
        shapelist = img_copy.shape

        emptyImage = img_copy.copy()
        pic = cv2.resize(emptyImage, (shapelist[0], shapelist[1]), interpolation=cv2.INTER_CUBIC)

        recognize_flag, embedded = normalization_img(pic)
        print("2 recognize_flag :",recognize_flag)

    if recognize_flag:
        print("擷取人臉特徵成功,開始辨識")


        example_prediction = knn.predict([embedded[1]])

        start_distance = int((len(embedded_metadata_database) / classnum) * (int(example_prediction)))
        End_distance = int((len(embedded_metadata_database) / classnum) * (int(example_prediction) + 1)) - 1


        for j in range(End_distance - start_distance + 1):
            Distance_Value = (distance(embedded_metadata_database[start_distance + j], embedded[1]))
            print("Distance_Value: ", Distance_Value)

            if Distance_Value <= distance_thresholde:
                 print(example_prediction)
                 example_identity = encoder.inverse_transform(example_prediction)[0]
                 print("辨識結果 :", example_identity)
                 response=example_identity
                 return json.dumps(response)


            elif j == (End_distance - start_distance):
                 print("辨識結果 : none")
                 response = "none"
                 return json.dumps(response)

    else:
        print("辨識結果 : exception")
        response = "exception"
        return json.dumps(response)


if __name__=='__main__':
    app.run('127.0.0.1', port=5000)
    # app.run( host='192.168.8.86',port=5000)

