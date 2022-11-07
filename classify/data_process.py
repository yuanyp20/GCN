import pandas as pd
import numpy as np
import csv
import os
from pandas import DataFrame
import math


# 将目录中所有文件的文件名，放入一个列表中
def list_file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            # 选择文件的格式
            if os.path.splitext(file)[1] == '.csv':
                L.append(os.path.join(root, file))
    return L


name5 = 'D:\oil0110\oil\HD5_378000'  # 378000
name20 = 'D:\oil0107\oil\HD20_94500'  # 94500


# # 将文件名列表中所有的文件，从xlsx格式转换成csv
# def xlsx_to_csv_pd(L):
#     for file_name in L:
#         data_xls = pd.read_excel(file_name, index_col=None, header=None, engine='openpyxl')
#         print('*************' + file_name + '************************')
#         print(data_xls.info())
#         data_xls.to_csv(file_name[:-4] + 'csv', encoding='utf-8')
#
# # 将xlsx转换成csv的示例
# list_20 = list_file_name(name20)
# print(list_20)
# xlsx_to_csv_pd(list_20)


# # 将数据多余部分切掉（378000之后的；94500之后的）
# def cut_data(filelist, data_len):
#     for file in filelist:
#         data = pd.read_csv(file, engine='python', index_col=0, header=0)
#         cut_len = [i for i in range(data_len, len(data))]
#         data = data.drop(cut_len)
#         data.to_csv(file)
#         print(file)
#
#
# l5 = list_file_name(name5)
# cut_data(l5, 378000)
# l20 = list_file_name(name20)
# cut_data(l20, 94500)

# def compress_and_cal_label(filelist, group_size, speed):
#     for file in filelist:
#         data = pd.read_csv(file, engine='python', header=0, index_col=0)
#         print(data.info())
#
#         for i in range(len(data) // group_size):
#             a = data.values[i * group_size:(i + 1) * group_size].mean(axis=0).tolist()  # 四舍五入，保留5位小数
#             # if speed == 5:
#             #     a.append(0.0402 * math.exp(0.007 * (20 + i * (1 / 12))))
#             # if speed == 20:
#             #     a.append(0.038 * math.exp(0.0068 * (20 + i * (1 / 3))))
#             if speed == 5:
#                 a.append(0.0402 * math.exp(0.007 * (20 + i * 5)))
#             # 5摄氏度/每分钟 RO=0.0402*e^0.007t
#             # 20摄氏度/每分钟 RO=0.038*e^0.0068t
#             a = DataFrame(a)
#
#             a.T.to_csv(file[:-4] + 'label.csv', mode='a', encoding='utf-8', header=False, index=False,
#                        float_format='%.6f')
#
#
# l5 = list_file_name(name5)
# compress_and_cal_label(l5, 3000, 5)
# l20 = list_file_name(name20)
# compress_and_cal_label(l20, 3000, 20)
