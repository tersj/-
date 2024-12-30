import os
import cv2
import numpy as np

img = cv2.imread(r'original_picture.jpg')  # 最终用于识别的图像


# 车牌定位
def license_region(image):
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    # 求出三种阈值
    license_region_thresh = np.zeros(np.append(3, r.shape))  # 创建一个空的三维数组用于存放三种阈值
    license_region_thresh[0, :, :] = r / b
    license_region_thresh[1, :, :] = g / b
    license_region_thresh[2, :, :] = b
    # 存放满足阈值条件的像素点坐标
    region_origin = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (license_region_thresh[0, i, j] < 0.35 and
                license_region_thresh[1, i, j] < 0.9 and
                license_region_thresh[2, i, j] > 90) or (
                    license_region_thresh[1, i, j] < 0.35 and
                    license_region_thresh[0, i, j] < 0.9 and
                    license_region_thresh[2, i, j] < 90):
                region_origin.append([i, j])
    region_origin = np.array(region_origin)
    # 进一步缩小行的索引范围
    row_index = np.unique(region_origin[:, 0])
    row_index_number = np.zeros(row_index.shape, dtype=np.uint8)
    for i in range(region_origin.shape[0]):
        for j in range(row_index.shape[0]):
            if region_origin[i, 0] == row_index[j]:
                row_index_number[j] = row_index_number[j] + 1
    row_index_out = row_index_number > 10  # 将误判的点去除
    row_index_out = row_index[row_index_out]
    # 进一步缩小列的索引范围
    col_index = np.unique(region_origin[:, 1])
    col_index_number = np.zeros(col_index.shape, dtype=np.uint8)
    for i in range(region_origin.shape[0]):
        for j in range(col_index.shape[0]):
            if region_origin[i, 1] == col_index[j]:
                col_index_number[j] = col_index_number[j] + 1
    col_index_out = col_index_number > 10
    col_index_out = col_index[col_index_out]
    # 得出最后的区间
    region_out = np.array([[np.min(row_index_out), np.max(row_index_out)],
                           [np.min(col_index_out), np.max(col_index_out)]])
    return region_out


region = license_region(img)
# 显示车牌区域
img_test = img.copy()  # 拷贝时不能直接等号赋值
cv2.rectangle(img_test, pt1=(region[1, 0], region[0, 0]), pt2=(region[1, 1], region[0, 1]),
              color=(0, 0, 255), thickness=2)
cv2.imshow('original_picture', img_test)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 分割区域灰度化、二值化
img_car_license = img[region[0, 0]:region[0, 1], region[1, 0]:region[1, 1], :]
img_car_license_gray = cv2.cvtColor(img_car_license, cv2.COLOR_BGR2GRAY)  # 将RGB图像转化为灰度图像
# otus二值化
img_car_license_binary = cv2.threshold(img_car_license_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow('bianry', img_car_license_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 车牌分割（均分割为25*15的图片）height=25,width=15
# 模板分割函数，只针对单个字符，用于去除其周围的边缘，并resize
def template_segmentation(origin_img):
    # 提取字符各列满足条件(有两个255的单元格)的索引
    col_index = []
    for col in range(origin_img.shape[1]):  # 对于图像的所有列
        if np.sum(origin_img[:, col]) >= 2 * 255:
            col_index.append(col)
    col_index = np.array(col_index)
    # 提取字符各行满足条件(有两个255的单元格)的索引
    row_index = []
    for row in range(origin_img.shape[0]):
        if np.sum(origin_img[row, :]) >= 2 * 255:
            row_index.append(row)
    row_index = np.array(row_index)
    # 按索引提取字符(符合条件的行列中取min-max)，并resize到25*15大小
    output_img = origin_img[np.min(row_index):np.max(row_index) + 1, np.min(col_index):np.max(col_index) + 1]
    output_img = np.uint8(output_img)
    if col_index.shape[0] <= 3 or row_index.shape[0] <= 3:
        output_img = origin_img[np.min(row_index):np.max(row_index) + 1, np.min(col_index):np.max(col_index) + 1]
        pad_row1 = np.int8(np.floor((25 - output_img.shape[0]) / 2))
        pad_row2 = np.int8(np.ceil((25 - output_img.shape[0]) / 2))
        pad_col1 = np.int8(np.floor((15 - output_img.shape[1]) / 2))
        pad_col2 = np.int8(np.ceil((15 - output_img.shape[1]) / 2))
        output_img = np.pad(output_img, ((pad_row1, pad_row2), (pad_col1, pad_col2)), 'constant',
                            constant_values=(0, 0))
        output_img = np.uint8(output_img)
    else:
        output_img = cv2.resize(output_img, (15, 25), interpolation=0)
    return output_img


# 对原始车牌抠图，抠出每一个字符
temp_col_index = []
# print(img_car_license_binary.shape[1])
for col in range(img_car_license_binary.shape[1]):
    # print(np.sum(img_car_license_binary[:, col]))
    if np.sum(img_car_license_binary[:, col]) >= 5 * 255:  # 提取大于等于5个255的列
        temp_col_index.append(col)
print(temp_col_index)
temp_col_index = np.array(temp_col_index)
flag = 0  # 值是7个字符的起始列
flag_i = 0  # 值的变化范围：从0到6(对应车牌的7个字符)
car_license_out_col = np.uint8(np.zeros([7, 30]))  # 7行的数组存储车牌上的7个需识别的字
for j in range(temp_col_index.shape[0] - 1):
    if temp_col_index[j + 1] - temp_col_index[j] >= 2:  # 提取的>=5个255的列之间不是相邻的(可初步解决川的分割问题)
        temp = temp_col_index[flag:j + 1]
        flag = j + 1
        if temp.shape[0] < 10:
            continue
        print(temp.shape[0])
        temp = np.append(temp, np.zeros(30 - temp.shape[0]))  # 补成30维的向量，方便最后赋值给car_license_out_col
        temp = np.uint8(temp.reshape(1, 30))
        car_license_out_col[flag_i, :] = temp
        flag_i = flag_i + 1
if flag_i < 7:
    temp = temp_col_index[flag:]
    temp = np.append(temp, np.zeros(30 - temp.shape[0]))
    temp = np.uint8(temp.reshape(1, 30))
    car_license_out_col[flag_i, :] = temp

# 分别提取7个字符
car_license_out_row = np.uint8(np.zeros([7, 30]))
for row in range(car_license_out_row.shape[0]):  # car_license_out_row.shape[0]
    temp = car_license_out_col[row, :]
    index = 0
    for i in range(temp.shape[0]):  # 去除列索引中多余的0
        if temp[i] == 0:
            index = i
            break
    col_temp = temp[0:index]
    temp_img = img_car_license_binary[:, np.min(col_temp):np.max(col_temp) + 1]
    t = np.nonzero(np.sum(temp_img, axis=1))
    if row == 0:
        province1 = temp_img[t, :]  # 汉字后续扩展成40*40
        province1 = province1[0, :, :]
        province1 = template_segmentation(province1)
        province1 = np.uint8(province1)
    if row == 1:
        province2 = temp_img[t, :]  # 字母和数字后续扩展成40*40
        province2 = province2[0, :, :]
        province2 = template_segmentation(province2)
        province2 = np.uint8(province2)
    if row == 2:
        car_number1 = temp_img[t, :]
        car_number1 = car_number1[0, :, :]
        car_number1 = template_segmentation(car_number1)
        car_number1 = np.uint8(car_number1)
    if row == 3:
        car_number2 = temp_img[t, :]
        car_number2 = car_number2[0, :, :]
        car_number2 = template_segmentation(car_number2)
        car_number2 = np.uint8(car_number2)
    if row == 4:
        car_number3 = temp_img[t, :]
        car_number3 = car_number3[0, :, :]
        car_number3 = template_segmentation(car_number3)
        car_number3 = np.uint8(car_number3)
    if row == 5:
        car_number4 = temp_img[t, :]
        car_number4 = car_number4[0, :, :]
        car_number4 = template_segmentation(car_number4)
        car_number4 = np.uint8(car_number4)
    if row == 6:
        car_number5 = temp_img[t, :]
        car_number5 = car_number5[0, :, :]
        car_number5 = template_segmentation(car_number5)
        car_number5 = np.uint8(car_number5)

cv2.imshow('province1', province1)
cv2.imshow('province2', province2)
cv2.imshow('car_number1', car_number1)
cv2.imshow('car_number2', car_number2)
cv2.imshow('car_number3', car_number3)
cv2.imshow('car_number4', car_number4)
cv2.imshow('car_number5', car_number5)
cv2.waitKey(0)
cv2.destroyAllWindows()


car_character = np.uint8(np.zeros([7, 25, 15]))
car_character[0, :, :] = province1.copy()
car_character[1, :, :] = province2.copy()
car_character[2, :, :] = car_number1.copy()
car_character[3, :, :] = car_number2.copy()
car_character[4, :, :] = car_number3.copy()
car_character[5, :, :] = car_number4.copy()
car_character[6, :, :] = car_number5.copy()


# 车牌识别
# 读取原始图片并生成模板的函数
def template_array_generator(template_path, template_size):
    template_img_out = np.zeros([template_size, 25, 15], dtype=np.uint8)
    index = 0
    files = os.listdir(template_path)
    for file in files:
        template_img = cv2.imdecode(np.fromfile(template_path + '/' + file, dtype=np.uint8), -1)
        if len(template_img.shape) < 3:
            template_img_gray = template_img
        else:
            template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        template_img_binary = cv2.threshold(template_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        template_img_out[index, :, :] = template_segmentation(template_img_binary)
        index = index + 1
    return template_img_out


Chinese_character = []
Chinese_char_template = []
Chinese_char_match_len = 0

Number_character = []
Number_char_template = []
Number_char_match_len = 0

Alphabet_character = []
Alphabet_char_template = []
Alphabet_char_match_len = 0

# template路径
templates_dir = 'template/refer1'
for filename in os.listdir(templates_dir):
    if 48 <= ord(filename) <= 57:
        Number_character.append(filename)
        number_path = os.path.join(templates_dir, filename)
        number_len = len(os.listdir(number_path))
        # print(number_path)
        number_template_out = template_array_generator(number_path, number_len)
        Number_char_template.append(number_template_out)
        Number_char_match_len = Number_char_match_len + number_template_out.shape[0]
    elif 65 <= ord(filename) <= 90:
        Alphabet_character.append(filename)
        alphabet_path = os.path.join(templates_dir, filename)
        alphabet_len = len(os.listdir(alphabet_path))
        alphabet_template_out = template_array_generator(alphabet_path, alphabet_len)
        Alphabet_char_template.append(alphabet_template_out)
        Alphabet_char_match_len = Alphabet_char_match_len + alphabet_template_out.shape[0]
    elif ord(filename) > 127:
        Chinese_character.append(filename)
        chinese_path = os.path.join(templates_dir, filename)
        chinese_len = len(os.listdir(chinese_path))
        chinese_template_out = template_array_generator(chinese_path, chinese_len)
        Chinese_char_template.append(chinese_template_out)
        Chinese_char_match_len = Chinese_char_match_len + chinese_template_out.shape[0]

# 进行字符识别
car_character = np.uint8(np.zeros([7, 25, 15]))
car_character[0, :, :] = province1.copy()
car_character[1, :, :] = province2.copy()
car_character[2, :, :] = car_number1.copy()
car_character[3, :, :] = car_number2.copy()
car_character[4, :, :] = car_number3.copy()
car_character[5, :, :] = car_number4.copy()
car_character[6, :, :] = car_number5.copy()
match_length = Chinese_char_match_len + Alphabet_char_match_len + Number_char_match_len
match_mark = np.zeros([7, match_length])

Chinese_char_start = 0
Chinese_char_end = Chinese_char_match_len
Alphabet_char_start = Chinese_char_match_len
Alphabet_char_end = Chinese_char_match_len + Alphabet_char_match_len
Number_char_start = Chinese_char_match_len + Alphabet_char_match_len
Number_char_end = match_length

chinese_index = 0
number_index = 0
alphabet_index = 0

for i in range(match_mark.shape[0]):  # 7个需识别的字符
    chinese_index = 0
    number_index = 0
    alphabet_index = 0
    for j in range(len(Chinese_char_template)):  # 所有的汉字模板
        for k in range(Chinese_char_template[j].shape[0]):
            match_mark[i, chinese_index + Chinese_char_start] = cv2.matchTemplate(car_character[i, :, :],
                                                             Chinese_char_template[j][k, :, :], cv2.TM_CCOEFF)
            chinese_index += 1
    # 所有的字母模板
    for j in range(len(Alphabet_char_template)):
        for k in range(Alphabet_char_template[j].shape[0]):
            match_mark[i, alphabet_index + Alphabet_char_start] = cv2.matchTemplate(car_character[i, :, :],
                                                              Alphabet_char_template[j][k, :, :],
                                                              cv2.TM_CCOEFF)
            alphabet_index += 1
    # 所有的数字模板
    for j in range(len(Number_char_template)):
        for k in range(Number_char_template[j].shape[0]):
            match_mark[i, number_index + Number_char_start] = cv2.matchTemplate(car_character[i, :, :],
                                             Number_char_template[j][k, :, :],
                                             cv2.TM_CCOEFF)
            number_index += 1
output_index = np.argmax(match_mark, axis=1)
output_char = []
for i in range(output_index.shape[0]):
    if Chinese_char_start <= output_index[i] <= Chinese_char_end:
        chinese_result = output_index[i] - Chinese_char_start
        for j in range(len(Chinese_char_template)):
            if chinese_result >= Chinese_char_template[j].shape[0]:
                chinese_result -= Chinese_char_template[j].shape[0]
            else:
                output_char.append(Chinese_character[j])
                break
    if Alphabet_char_start <= output_index[i] <= Alphabet_char_end:
        alphabet_result = output_index[i] - Alphabet_char_start
        for j in range(len(Alphabet_char_template)):
            if alphabet_result >= Alphabet_char_template[j].shape[0]:
                alphabet_result -= Alphabet_char_template[j].shape[0]
            else:
                output_char.append(Alphabet_character[j])
                break
    if Number_char_start <= output_index[i] <= Number_char_end:
        number_result = output_index[i] - Number_char_start
        for j in range(len(Number_char_template)):
            if number_result >= Number_char_template[j].shape[0]:
                number_result -= Number_char_template[j].shape[0]
            else:
                output_char.append(Number_character[j])
                break

# 打印识别结果
for i in range(len(output_char)):
    if i == 0:
        print('province1:' + output_char[0])
    if i == 1:
        print('province1:' + output_char[1])
    if i == 2:
        print('car1:' + output_char[2])
    if i == 3:
        print('car2:' + output_char[3])
    if i == 4:
        print('car3:' + output_char[4])
    if i == 5:
        print('car4:' + output_char[5])
    if i == 6:
        print('car5:' + output_char[6])
