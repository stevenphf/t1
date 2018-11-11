import cv2,sys
import numpy as np
import tensorflow as tf

def pic2data(img_name):
    # 获取图片
    img = cv2.imread(img_name)
    #转换成灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #去噪（有 均值滤波器、高斯滤波器、中值滤波器、双边滤波器等, 高斯去噪效果最好）
    #blurred = cv2.GaussianBlur(gray, (9, 9),0)
    #反转
    _, binary_inv = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    #找区域轮廓 findContours函数
    #参数一：要检索的图片，必须为二值化图像，即黑白的（不是灰度图）
    #参数二：轮廓类型
    #   cv2.RETR_EXTERNAL           表示只检测外轮廓
    #   cv2.RETR_CCOMP              建立两个等级的轮廓,上一层是外边界，里一层为内孔边界信息
    #   cv2.RETR_LIST               检测的轮廓不建立等级关系
    #   cv2.RETR_TREE               建立一个等级树结构的轮廓
    #参数三：处理近似方法
    #   cv2.CHAIN_APPROX_NONE       存储所有的轮廓点，相邻的两个点的像素位置差不超过1
    #   cv2.CHAIN_APPROX_SIMPLE     压缩垂直、水平、对角方向，只保留端点
    #   cv2.CHAIN_APPROX_TC89_L1    使用teh-Chini近似算法
    #   cv2.CHAIN_APPROX_TC89_KCOS  使用teh-Chini近似算法
    (_, cnts, _) = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #计算矩阵轮廓
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))  #4点坐标矩阵

    #画轮廓
    draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
    #据轮廓切割图
    #img_crop = img[y1_min:y2_max,x1_min:x2_max,:]
    #img_crop = gray[12:542,120:577]  #按照未旋转的轮廓截取，旋转待研究，目前没有必要
    y_min = box.T[1].min()
    y_max = box.T[1].max()
    x_min = box.T[0].min()
    x_max = box.T[0].max()
    print(y_min, y_max, x_min, x_max)
    p=0.4
    y_temp = (int(y_max) - int(y_min))*p/2
    x_temp = (int(x_max) - int(x_min))*p/2
    y_min = max(0,y_min-y_temp)
    y_max = y_max + y_temp/2
    x_min = max(0, x_min - x_temp)
    x_max = x_max + x_temp
    print(y_min, y_max, x_min, x_max)
    print(int(y_min), int(y_max), int(x_min), int(x_max))
    img_crop = binary_inv[int(y_min):int(y_max),int(x_min):int(x_max)]

    #图像缩小
    res = cv2.resize(img_crop,(28,28),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("temp_res.jpg", res)
    #全部显示
    #cv2.imshow("img", img)
    #cv2.imshow("gray", gray)
    #cv2.imshow("binary_inv", binary_inv)
    #cv2.imshow("img_crop", img_crop)
    cv2.imshow(str(img_name)+"-draw_img", draw_img)
    cv2.imshow(str(img_name)+"-res", res)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #转换成可对接的数据
    res.shape = -1,784
    x = res.astype(np.float32)
    xr = np.multiply(x,1.0/255)
    #print('res.shape,xr.shape:\t'+str(res.shape)+',\t'+str(xr.shape))
    return xr

# ----------------------------------------------------------------------------------------

def tensor_flow(x_value, prob_value):
    with tf.Session() as sess:
        # 取Model
        sess.run(init)
        # add by steven
        #writer = tf.summary.FileWriter("/home/lab/tf/tp_log", sess.graph)

        saver.restore(sess, '/home/stevenpan/work/tf/ModelconvNN2/model.ckpt')
        y_result = sess.run(y_conv, feed_dict={x: x_value, keep_prob: prob_value})
        # label_result = tf.argmax(y_result,1)
        # print('结果：' + str(np.argmax(y_result)))
    #writer.close()
    return str(np.argmax(y_result))


#import tensorflow as tf
#建模型
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])
    keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x,[-1,28,28,1])    #转成4d张量
#定义处理函数
def conv2d(x,w):
    #卷积不改变输入的shape
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#对Tensorflow的池化进行封装
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def weight_variable(shape):
    #初始化权重，正态分布，标准方差为0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    #初始化偏置值，设置非零避免死神经元
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

with tf.name_scope('Inference'):
    W_conv1 = weight_variable([3,3,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([6,6,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])

    w_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    predictions = tf.nn.softmax(y_conv)

saver = tf.train.Saver()
init = tf.global_variables_initializer()


#---main ------
imgs = sys.argv[1:]
r = []
t_name = ''
if len(imgs)>0:
    for img in imgs:
        temp = tensor_flow(pic2data(img),1)
        r.append(temp)
        if t_name == '':
            t_name = img
        else:
            t_name= t_name + '-' + img
        print(img+'：\t' + temp)
    print('程序对' + t_name + '的识别为：\t' + '-'.join(list(r)))
else:
    print('t.jpg：\t' + tensor_flow(pic2data('t.jpg'), 1))
