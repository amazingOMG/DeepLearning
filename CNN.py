#宜蘭大學 資工二 B1043201 鄧秀鳳
from keras.datasets import mnist  # 使用套件匯入MNIST資料集
from keras.utils import np_utils  # 後續將 label標籤 轉換成One-hot encoding需要
from keras.models import Sequential  # 建立訓練模型需要此套件
from keras.layers import Conv2D , MaxPooling2D , Flatten , Dense , Dropout # 卷積層、池化層、平坦層、全連接層、Dropout層
import numpy as np # 矩陣相關運算需要此套件
import matplotlib.pyplot as plt  # 將資料視覺化，可用圖表呈現結果
import pandas as pd # 混淆矩陣需要此套件

def preprocessing():  # 前置處理
    global image_Train , image_Test

    image_Train = image_Train.reshape( image_Train.shape[0] , 28 , 28 , 1 ).astype( 'float32' )  # 將二維圖片資料( 28 * 28 = 784 )轉換為( 60000 * 28 * 28 * 1 )的四維矩陣資料
    image_Test = image_Test.reshape( image_Test.shape[0] , 28 , 28 , 1 ).astype( 'float32' )

    image_Train /= 255  # 黑白圖的像素值介於0~255之間，除以255將特徵值壓縮至0與1之間(提升模型預測的準確度 。 梯度運算時加速收斂 )
    image_Test /= 255

def One_hot( classification ):  # 將標籤轉化成 One-hot encoding 的編碼格式
    label_Train_OneHot = np_utils.to_categorical( label_Train , classification )
    label_Test_OneHot = np_utils.to_categorical( label_Test , classification )
    return label_Train_OneHot , label_Test_OneHot

def Convolution_Layer( filters ): # 卷積層
    model.add(  Conv2D( filters , kernel_size = ( 5 , 5 ) , padding = 'same' , input_shape = ( 28 , 28 ,  1 ) , activation = 'relu' )  )

def Pooling_Layer(): # 池化層
    model.add( MaxPooling2D(  pool_size = ( 2 , 2 )  ) )

def layers( classification ):  # 設置輸入層、隱藏層、輸出層、Dropout層、平坦層
    Convolution_Layer( 16 )
    Pooling_Layer()
    Convolution_Layer( 32 )
    Pooling_Layer()

    model.add( Dropout( 0.5 ) )
    model.add( Flatten() ) # 平坦層
    model.add(  Dense( 128 , activation = 'relu' )  )
    model.add( Dropout( 0.5 ) )

    model.add(  Dense( classification , activation = 'softmax' )  ) # 輸出層

def diagrams( train , test ):  # 繪製圖表
    plt.figure(  figsize = ( 6 , 4 )  )
    plt.plot( train_History.history[ train ] , label = 'Train ' + train )
    plt.plot( train_History.history[ test ] , label = 'Test ' + train )
    plt.title( 'Train And Test ' + train )
    plt.ylabel( train + ' Probability' )
    plt.xlabel( 'Epoch' )
    plt.legend()
    plt.show()

def Label_Image( label , image , predict , index ):
    plt.figure(  figsize = ( 12 , 14 )  ) # 設置圖形大小

    for loop in range( 0 , 10 ):
        ax = plt.subplot( 5 , 5 , 1 + loop )
        ax.imshow( image[ index[loop] ] , cmap = 'binary' )

        title = 'label = %s , predict = %s ' %( str( label[ index[loop] ] ) , str( predict[ index[loop] ] ) ) # 顯示標籤欄位

        ax.set_title( title , fontsize = 10 ) # 設定 image 與 title 的大小
        ax.set_xticks( [] ) # 設定不顯示x軸刻度
        ax.set_yticks( [] ) # 設定不顯示y軸刻度

    plt.show()

########################################################################################################################################

print( ' CNN 識別 MNIST 數字資料集 ' )
np.random.seed( 10 )
( image_Train , label_Train ) , ( image_Test , label_Test ) = mnist.load_data()  # 載入MNIST資料集
#          訓練集                            測試集
#       ( 圖片 , 標籤 )                   ( 圖片 , 標籤 )

print( 'Train data = %d 筆資料' % ( len( image_Train ) ) )  # 輸出訓練集總數
print( 'Test data = %d 筆資料\n\n' % ( len( image_Test ) ) )  # 輸出測試集總數

nb_classes = 10  # 數字圖片為 0~9，共有10種類別
preprocessing() # 進行前置處理

OneHot_Train , OneHot_Test = One_hot( nb_classes ) # 將標籤轉換為 One-hot 編碼

################################################### 以上為前置處理 #####################################################################

model = Sequential() # 定義模型：宣告優化器
layers( nb_classes ) # 設置輸入層、隱藏層、輸出層、Dropout層、平坦層
model.summary()  # 顯示模型摘要資訊
# Layer (type) 顯示每個層集的類別    Output Shape：輸出尺寸   Param：每個層集神經元的權重數量

################################################### 以上為定義模型 #####################################################################

model.compile( loss = 'categorical_crossentropy' , optimizer = 'SGD' , metrics = ['accuracy'] )
# compile：編譯模型    loss：損失函數(crossentropy為交叉熵)     optimizer：優化器(SGD：隨機梯度下降法)  metrics：模型評估方式(accutacy：以準確度為主)

################################################### 以上為編譯模型設定 #################################################################

print( '\n\n' )
epochs = 10  # epochs：訓練週期( 1個epoch = 16萬筆資料訓練過1次 ) ，不可定義太高，容易過擬合
train_History = model.fit( image_Train , OneHot_Train , validation_split = 0.2 , epochs = epochs , batch_size = 128 , verbose = 2 )
# batch_size：每次丟多少資料進行訓練   verbose：訓練日誌，顯示模型(可觀察loss損失函數以及accuracy準確度的變化 )

################################################### 以上為訓練模型 #####################################################################

print( '\n\n' )
diagrams( 'accuracy' , 'val_accuracy' )  # 傳入：訓練集的準確率 與 測試集的準確率
diagrams( 'loss' , 'val_loss' )  # 傳入：訓練集的誤差率 與 測試集的誤差率

################################################### 以上為資料視覺化 ###################################################################

loss , accuracy = model.evaluate( image_Train , OneHot_Train , verbose = 2 )  # 評估模型準確率
print(  '訓練集的準確度 = {:.2f}%\n'.format( accuracy * 100 )  )
loss , accuracy = model.evaluate( image_Test , OneHot_Test , verbose = 2 )  # 評估模型準確率
print(  '測試集的準確度 = {:.2f}%\n\n'.format( accuracy * 100 )  )

################################################### 以上為評估模型 #####################################################################

prediction = np.argmax( model.predict( image_Test ) , axis = -1 ) # 預測測試集內的數字影像資料
print( '測試集內 的 數字影像資料 預測結果為：' + str( prediction ) + '\n\n' ) # 輸出預測結果
confusion_Matrix = pd.crosstab( label_Test , prediction , rownames = [ 'label' ] , colnames = [ 'predict' ] ) # 建立混淆矩陣
print( str ( confusion_Matrix ) )

################################################# 以上為建立混淆矩陣 ###################################################################

dataFrame = pd.DataFrame( { 'label' : label_Test , 'predict' : prediction } ) # 將數字影像標籤與預測結果匯入表格

predict_Index = []
for loop in range( 0 , 10 ):
    print( '\n\n根據混淆矩陣，輸入待查詢的預測值    label = %d ， predict = ' %( loop ) , end = '' )
    predict_Number = int( input() )
    check = dataFrame[ ( dataFrame.label == loop ) & ( dataFrame.predict  == predict_Number ) ]
    print( '真實值是 %d ， 但預測值是 %d 的資料：\n' %( loop , predict_Number ) + str( check.head() ) )
    print( '輸入想查詢的索引值：' , end = '' )
    index = int( input() )
    predict_Index.append( index )

Error = dataFrame[ prediction != label_Test ] # 將預測數字與影像數字不符的結果製成表格
print( '\n\n' + str( Error ) ) # 輸出表格

Label_Image( label_Test , image_Test , prediction , predict_Index ) # 將此預測結果與數字影像、數字標籤以圖形呈現

################################################# 以上為預測模型結果 ####################################################################
