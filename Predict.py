

import os.path
import gc 
from sklearn.externals import joblib  # jbolib模組
from Load_Metadata import load_metadata,normalization,label,distance,recordToCSV,move,copyimg,load_metadata_predict
    

#set path
path_database='./images/'
path_recognize='./Taipei/Check'
path_recognize_folder = './Taipei/Check/folder'
path_finish = './Taipei/RecognizePhoto'
path_finish2 ='./Taipei/RecognizePhoto2'


classnum = len(os.listdir(path_database)) #資料庫有幾個人
print("classnum: "  ,classnum)
distance_thresholde=0.5

#load model
svc = joblib.load('models/svc.pkl')
#set flag
RecognizeFlag = 1

# prepare before recognize
# 讀取database 照片
metadata_database = load_metadata(path_database)
# 原始圖片正規化
recognize_flag_metadata_database, embedded_metadata_database,i_metadata_database = \
    normalization(metadata_database)
encoder = label(metadata_database)


print("start")

while 1:
    
    gc.collect()
    if len(os.listdir(path_recognize_folder)) != 0:
            
            Photo_name=os.listdir(path_recognize_folder)
            print("檔案名稱:",Photo_name)
            
            metadata_recognize_folder = load_metadata_predict(path_recognize)    
            recognize_flag, embedded,i = normalization(metadata_recognize_folder)

            # print("recognize_flag :",recognize_flag)

            if recognize_flag == 0:
                print("擷取人臉特徵失敗")
                copyimg(path_recognize_folder, Photo_name, path_finish2)
                

                metadata_recognize_folder = load_metadata_predict(path_recognize)
                recognize_flag, embedded, i = normalization(metadata_recognize_folder)
            
    else:
        continue
        time.sleep(0.1)

    if recognize_flag:
        print("擷取人臉特徵成功,開始辨識")
 
        #如果有多種圖要辨識的話
        example_idx = i
        example_prediction = svc.predict([embedded[example_idx]])

        start_distance=int((len(embedded_metadata_database)/classnum) * (int(example_prediction)))
        End_distance = int((len(embedded_metadata_database)/classnum) * (int(example_prediction)+1))-1

        Total_distance=[]
        
        for j in range(End_distance-start_distance+1):
            Distance_Value = (distance(embedded_metadata_database[start_distance+j], embedded[example_idx]))
            print("Distance_Value: ", Distance_Value)
           
            
            if Distance_Value <=  distance_thresholde:
               # print("min", min(Total_distance ))
                print(example_prediction)
                example_identity = encoder.inverse_transform(example_prediction)[0]
                print("辨識結果 :",example_identity)

                recordToCSV(Photo_name[0], example_identity)
                break
            
            elif j==(End_distance-start_distance):
                    recordToCSV(Photo_name[0] ,"none")
                    print("辨識結果 : none")
                    break
    else:
        recordToCSV(Photo_name[0], "exception")
        print("辨識結果 : exception")

    move(path_recognize_folder, path_finish,Photo_name[0])
    
    del Photo_name,metadata_recognize_folder,recognize_flag, embedded, i   
    gc.collect()
    
    


