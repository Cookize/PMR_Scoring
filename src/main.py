#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab 
import scipy.stats as stats
import math


# In[2]:


# 导入数据
def load_data():
    print("/----------------------------------------------------/")
    print("Start loading data:")
    data = pd.read_excel(open('../data/only-score.xlsx', 'rb'))
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    print("Finish loading data!\nShape of data:", data.shape)
    return data


# In[3]:


# 根据缺失数据位置进行切片，得到无缺失的评分矩阵
def drop_all_na(df):
    print("/----------------------------------------------------/")
    print("Start droping all 'nan' values:")
    if not isinstance(df, pd.DataFrame):
        print("ERROR: Not DF.")
        return df
    ret_df = df.copy()
    ret_df.dropna(axis=1, how="all", inplace=True)
    ret_df.dropna(axis=0, how="any", inplace=True)
    print("Finish droping!\nShape of data:", ret_df.shape)
    return ret_df


# In[4]:


# 丢弃完全缺失的数据，得到部分缺失的评分矩阵
def drop_partail_na(df):
    print("/----------------------------------------------------/")
    print("Start droping partial 'nan' values:")
    if not isinstance(df, pd.DataFrame):
        print("ERROR: Not DF.")
        return df
    ret_df = df.copy()
    ret_df.dropna(axis=1, how="all", inplace=True)
    ret_df.dropna(axis=0, how="all", inplace=True)
    print("Finish droping!\nShape of data:", ret_df.shape)
    return ret_df


# In[5]:


# 打印评分Series的柱状图
def print_bar(data):
    plt.figure()
    plt.bar(data.index, data)
    plt.show()
    return
def print_dup_bar(data_list):
    total_width, n = 1, len(data_list) + 1
    width = total_width / n
    for i in range(len(data_list)):
        plt.bar(data_list[i].index + width * i, data_list[i], width=width)
    plt.show()


# In[6]:


# 检查是否对自己评分出现误差
def solve_selfish(data):
    print("/----------------------------------------------------/")
    print("Start checking selfishness:")
    self_score = pd.Series(data=data.columns.isin(data.index), index=data.columns)
    for column in data.columns:
        if not self_score[column]:
            continue
        if data.at[column, column] == np.nan:
            continue
        mean = data.loc[:,column].drop(column).mean(skipna=True)
        std = data.loc[:,column].drop(column).std(skipna=True)
        conf_intveral = stats.norm.interval(0.9, loc=mean, scale=std)
        if data.at[column, column] > conf_intveral[1]:
            data.at[column, column] = mean
            print(column, "is selfish.")
        elif data.at[column, column] < conf_intveral[0]:
            data.at[column, column] = mean
            print(column, "is lack of confidence.")
    normalize_score(data)
    print("Finish tuning!")
    return data


# In[7]:


# 按行/列标准化数据
def normalize_score(data, axis=0):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        mean = data.mean(axis=1-axis, skipna=True)
        std = data.std(axis=1-axis, skipna=True)
        data = data.sub(mean, axis=axis).div(std, axis=axis)
    return data


# In[8]:


# 按照样本反标准化数据
def denormalize_score(data, example):
    output = data.mean(axis=0, skipna=True).mul(example.std(skipna=True)).add(example.mean(skipna=True))
    return output


# In[9]:


# 使用减小评分人评分标准差距的策略
def get_score(score_df_origin, cnsidr_nan = False):
    
    if cnsidr_nan:
        score_df_lite = drop_partail_na(score_df_origin)
    else:
        score_df_lite = drop_all_na(score_df_origin)
    
    # 计算源数据的数值特征
    score_mean_origin = score_df_lite.mean(axis=0, skipna=cnsidr_nan)
    
    # 标准化源数据
    score_df_lite_norm = normalize_score(score_df_lite, axis=0)
    
    # 计算最终成绩 Version 1
    score_final_v1 = denormalize_score(score_df_lite_norm, score_mean_origin)
    
    # 减轻自私因素造成的偏差
    score_df_lite_norm_v2 = solve_selfish(score_df_lite_norm)
    
    # 计算最终成绩 Version 2
    score_final_v2 = denormalize_score(score_df_lite_norm_v2, score_mean_origin)
    
    return score_mean_origin.round(decimals=2), score_final_v1.round(decimals=2), score_final_v2.round(decimals=2)


# In[10]:


# 比较处理前后分数
def evaluate(data_list, index):
    print("/----------------------------------------------------/")
    print("Start evaluating result:")
    # 输出柱状图
    print_dup_bar(data_list)
    
    # 输出表格
    print("Comparing score:")
    print(pd.DataFrame(data=data_list, index=index).T)
    print("/----------------------------------------------------/")
    # 输出排名变化情况
    print("Comparing rank:")
    rank_list = [data.rank(ascending=False, method='min').astype('int') for data in data_list]
    print(pd.DataFrame(data=rank_list, index=index).T)


# In[11]:


if __name__ == "__main__":
    score_df = load_data()
    print("/----------------------------------------------------/")
    print("Solution 1: Drop all missing data:")
    score_origin, score_final_v1, score_final_v2 = get_score(score_df, False)
    evaluate([score_origin, score_final_v1, score_final_v2], ["Origin", "Normalization", "Deselfish"])
    print("/----------------------------------------------------/")
    plt.scatter(score_origin.index, score_origin.loc[:], s=50, label = '$Origin$', c = 'lightskyblue', marker='.', alpha = None, edgecolors= 'skyblue')
    plt.scatter(score_final_v1.index, score_final_v1.loc[:], s=50, label = '$Normalization$', c = 'springgreen', marker='.', alpha = None, edgecolors= 'mediumspringgreen')
    plt.scatter(score_final_v2.index, score_final_v2.loc[:], s=50, label = '$Final$', c = 'lightsalmon', marker='.', alpha = None, edgecolors= 'salmon')
    plt.legend()
    plt.show()
    
    print("/----------------------------------------------------/")
    print("Solution 2: Drop partial missing data:")
    score_origin, score_final_v1, score_norm_v2 = get_score(score_df, True)
    evaluate([score_origin, score_final_v1, score_final_v2], ["Origin", "Normalization", "Deselfish"])
    print("/----------------------------------------------------/")
    plt.scatter(score_origin.index, score_origin.loc[:], s=50, label = '$Origin$', c = 'lightskyblue', marker='.', alpha = None, edgecolors= 'skyblue')
    plt.scatter(score_final_v1.index, score_final_v1.loc[:], s=50, label = '$Normalization$', c = 'springgreen', marker='.', alpha = None, edgecolors= 'mediumspringgreen')
    plt.scatter(score_final_v2.index, score_final_v2.loc[:], s=50, label = '$Final$', c = 'lightsalmon', marker='.', alpha = None, edgecolors= 'salmon')
    plt.legend()
    plt.show()

