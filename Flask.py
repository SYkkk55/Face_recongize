import os.path
from sklearn.externals import joblib
import json
from Load_Metadata import load_metadata,normalization,normalization_Flask,label,\
    predic_Flask,img_resize_Flask,DatabaseModel
from model import create_model
print("set path--------------------")

print("set nn4 models---------------------")
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('models/nn4.small2.v1.h5')
distance_thresholde = 0.3

# Taipei = DatabaseModel('1','./images/Taipei/','models/knn_Taipei.pkl')
# Tainan = DatabaseModel('2','./images/Tainan/','models/knn_Tainan.pkl')

Taipei = DatabaseModel('1','./images/Taipei/','models/knn_Taipei.pkl')
Tainan = DatabaseModel('2','./images/Tainan/','models/knn_Tainan.pkl')

print("start")
#--------------------------------------------

import base64, json
import numpy as np
from flask import Flask, request
from flask_cors import CORS
import  cv2
app=Flask(__name__)
CORS(app, resources=r'/*')




@app.route('/', methods=['GET', 'POST'])
def root():

    # 從 flask request 中讀取圖片（byte str）
    image=request.form['image'].replace(' ', '+')
    filename = request.form.get("filename")
    type = request.form.get("type")

    print(filename)
    print("type",type)

    # 做 base64 的解碼
    image=base64.b64decode(image)
    img_array = np.fromstring(image, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print("-----接收到圖片-------")

    recognize_flag, embedded = normalization_Flask(img)

    if recognize_flag == 0:
        print("第一次擷取人臉特徵失敗")

        recognize_flag, embedded=img_resize_Flask(img)

    if recognize_flag:
        print("擷取人臉特徵成功,開始辨識")
        print(type)
        if int(type)==1:
            print("go to type 1")
            response=predic_Flask(Taipei.model,Taipei.embedded_metadata_database,Taipei.classnum,distance_thresholde,Taipei.encoder,embedded)
        elif int(type)==2:
            print("go to type 2")
            response = predic_Flask(Tainan.model, Tainan.embedded_metadata_database, Tainan.classnum,
                                    distance_thresholde, Tainan.encoder, embedded)


    else:
        response = "exception"
        print("辨識結果 : exception")

    print("response :", response)
    return json.dumps(response)


if __name__=='__main__':
    app.run('127.0.0.1', port=5000)
    # app.run( host='192.168.8.86',port=5000)

