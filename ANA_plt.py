# -*- coding: utf-8 -*-

import math
import time
import warnings
import pandas as pd
import numpy as np
import openpyxl as opy
import matplotlib.mlab as mlab
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy import special


def app_screen(x, down, up):
    if x < down or x > up:
        return np.NaN
    else:
        return x


def screen(data, down, up):
    down = data.quantile(down / 100)
    up = data.quantile(up / 100)
    empty = pd.DataFrame()
    n = 0
    for i in data.columns:
        n1 = data[i].index[0]
        if isinstance(data[i].loc[n1], str):
            pass
        else:
            screen_box = data[i].apply(app_screen, args=(down[i], up[i]))
            empty.insert(n, i, screen_box)
            n = n + 1
    return empty


def func(x, a, b, u, o):
    return a * (b + special.erf((x - u) / o * 2 ** 0.5))


def func_pdf(x, a, u, o):
    return a * (1 / (2 * np.pi * o ** 2) ** 0.5) * np.exp(-(x - u) ** 2 / (2 * o ** 2))


def func_pdf_find_x(y, m, n, num, a, u, o):
    xl = np.linspace(m, n, num, endpoint=True)
    yl = [a * (1 / (2 * np.pi * o ** 2) ** 0.5) * np.exp(-(xin - u) ** 2 / (2 * o ** 2)) for xin in xl]
    final_yl = [abs(yin - y) for yin in yl]
    out = pd.DataFrame({'x': xl, 'y': final_yl})
    return out


# 参数

pd.set_option('precision', 2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------- 连接 -----------

# excel 内只能包含表格，要把表格外的东西删掉（比如 数据来源：wind）

addr = r'C:\Users\Administrator\Desktop\ANA'
addr_in = addr + '\原始表.xlsx'
addr_out = addr + '\数值计算表.xlsx'
addr_out_pro = addr + '\函数拟合表.xlsx'
df = pd.read_excel(addr_in, sheet_name='对数')
standard = '三级行业'

# ----------- 数据分组与指标生成 -----------

sample_num = []
trans_stat = []
industry_list = df.drop_duplicates(['三级行业'])['三级行业'].tolist()

# 样本数目生成

for i in industry_list:
    df_trans = df[df[standard] == i]
    sample_num.append(len(df_trans))

# 插入样本数目后排序

s_n = pd.Series(sample_num, index=industry_list)
s_n_df = s_n.to_frame().reset_index()
s_n_df.columns = ['三级行业', '样本数']
s_n_df.sort_values(by='样本数', ascending=0, inplace=True)

# ----------- 数据分析 -----------

df_statpara = pd.read_excel(addr_out, sheet_name='均值')
statpara = df_statpara.columns[2:].tolist()
df_mean, df_svar = pd.DataFrame(), pd.DataFrame()
for stat_p in statpara:
    print(stat_p)

    # ----------- 分布曲线及拟合 -----------

    # 参数设置

    para = stat_p
    num = 49
    np.set_printoptions(precision=2)

    # 拟合

    n = 0
    dict_para = {}
    index_s = s_n_df['三级行业'][:num].tolist()

    # ----------- 累计分布函数 -----------

    for indust in index_s:
        n = n + 1

        # 数据导入

        a = df[df['三级行业'] == indust]
        b = a[['证券代码', para]]
        b = b.sort_values(by=para, ascending=1)
        x = b[para][b[para].notnull()].tolist()

        y = np.arange(0, len(x))

        # 累计分布情况

        plt.figure(para, figsize=(12, 12), dpi=60)
        plt.subplot(num ** 0.5, num ** 0.5, n)
        plt.title(indust)
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97, wspace=0.2, hspace=0.27)
        plt.plot(x, y, '--')

        # 累计分布函数拟合

        if stat_p.find('利') >= 0:
            orgin_para = [40, 2, -3, 0.5]
        else:
            orgin_para = [50, 1, -1, 0.5]
        if len(y) != 0:
            popt, pcov = curve_fit(func, x, y, orgin_para, maxfev=100000)
            y_fit = func(x, popt[0], popt[1], popt[2], popt[3])
            plt.plot(x, y_fit, '-')
        dict_para[indust] = popt = np.append(popt[:4], [b.min()[1], b.max()[1]])

        minmaxdict = dict()
        minmaxdict[indust] = [min(x), max(x)]

    # ----------- 密度分布函数 -----------

    n = 0
    dict_para_2 = {}
    for i, v in dict_para.items():

        # 求密度分布函数

        n = n + 1
        plt.subplot(num ** 0.5, num ** 0.5, n)

        if v[4] is None:
            v[4], v[5] = -7, 3

        X = np.linspace(v[4], v[5], 512, endpoint=True)
        C_l = []
        for i in X:
            C = func(i, v[0], v[1], v[2], v[3])
            C_l.append(C)
        d_l = []
        for i in range(0, len(X) - 1):
            d = (C_l[i + 1] - C_l[i]) / (X[i + 1] - X[i])
            d_l.append(d)
        x_l = X.tolist()[0: 511]
        plt.plot(x_l, d_l, '--')
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97, wspace=0.2, hspace=0.27)

        # 密度分布函数拟合

        popt, pcov = curve_fit(func_pdf, x_l, d_l, maxfev=100000)
        y_fit = func_pdf(x_l, popt[0], popt[1], popt[2])
        plt.subplot(num ** 0.5, num ** 0.5, n)
        plt.plot(x_l, y_fit, '--')
        dict_para_2[n] = popt

    # 参数输出

    df1 = pd.DataFrame(dict_para).T
    df1.columns = ['a', 'b', 'u', 'o', 'min', 'max']
    df2 = pd.DataFrame(dict_para_2).T
    df2.columns = ['a', 'u', 'o']
    df2.index = index_s
    c_num = len(df_mean.columns)
    df_mean.insert(c_num, stat_p, df2['u'])
    df_svar.insert(c_num, stat_p, df2['o'])
    plt.savefig(addr + '\图形\{}.png'.format(stat_p))

# excel 结果输出

fisrt_c = df_statpara['样本数'].tolist()[0:49]
df_mean.insert(0, '样本数', fisrt_c)
df_svar.insert(0, '样本数', fisrt_c)
write = pd.ExcelWriter(addr_out_pro)
df_mean.to_excel(write, sheet_name='均值')
df_svar.to_excel(write, sheet_name='标准差')
write.save()
write.close()
