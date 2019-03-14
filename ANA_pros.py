# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import warnings


def test(x):
    print('类型：\n{}\n'.format(type(x)))
    if isinstance(x, pd.Series):
        print('竖标：\n{}\n'.format(x.index))
    else:
        print('竖标：\n{}\n'.format(x.index))
        print('横标：\n{}\n'.format(x.columns))
    # print('内容：\n{}\n'.format(x.values))
    print('------------------------------------\n')


def func_pdf(x, a, u, o):
    return a * (1 / (2 * np.pi * o ** 2) ** 0.5) * np.exp(-(x - u) ** 2 / (2 * o ** 2))


def choose_right(x, u):
    if x < u:
        x = u
        return x


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ----------- 连接 -----------

# excel 内只能包含表格，要把表格外的东西删掉（比如 数据来源：wind）

addr = r'C:\Users\Administrator\Desktop\ANA'
addr_in = addr + '\原始表.xlsx'
addr_out = addr + '\函数拟合表.xlsx'
addr_final = addr + '\概率计算表.xlsx'

df_origin = pd.read_excel(addr_in, sheet_name='对数')
stata_list = ['均值', '标准差']
result_list = []
for i in stata_list:
    df_stata = pd.read_excel(addr_out, sheet_name=i)
    df_stata.rename(columns={'Unnamed: 0': '三级行业'}, inplace=True)
    result = pd.merge(df_origin, df_stata, on='三级行业', how='left')
    num = len(df_stata.columns) + 3
    b = result[result.columns[num:]]
    b.columns = [item.strip('_y') for item in b.columns]
    result_list.append(b)

a = df_origin[df_origin.columns[4:]]
b = result_list[0]
a[a < b] = -100000000000

# 计算指标

c = func_pdf(a, 1, result_list[0], result_list[1])
c = c[df_stata.columns[2:]]
c.to_csv(addr_final, encoding='utf-8-sig')
c.replace(0, inplace=True)
product = []
for i, v in c.iterrows():
    trans = v.sort_values().tolist()
    product_trans = trans[0] * trans[1] * trans[2]
    product.append(product_trans)
product_trans = pd.DataFrame(product)
c = pd.concat([c, product_trans], axis=1)
c.rename(columns={0: '指标'}, inplace=True)
print(c)

# 输出

write = pd.ExcelWriter(addr_final)
c.to_excel(write, sheet_name='概率')
write.save()
write.close()
