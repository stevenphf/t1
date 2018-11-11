import cv2,sys
import pandas as pd

lines = sys.argv[1:]

train = pd.read_csv('train.csv')

for line in lines:
    if int(line)>1:
        x_train = train.iloc[int(line)-2, 1:].values  #实际上取标号为line的行
        y_label = train.iloc[int(line)-2, :].values[0]
        x_train.shape = -1, 28
        cv2.imwrite("temp.jpg", x_train)
        img = cv2.imread("temp.jpg")
