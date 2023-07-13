# from modules.lookup import *

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from matplotlib import rc

from functools import partial
import multiprocessing as mp
import pandas as pd
import os
import openpyxl
from tqdm import trange
import glob
from os.path import join
import re
from multiprocessing import Pool, cpu_count
import numpy as np
from nltk import word_tokenize, bigrams
import swifter
import networkx as nx
from pyvis.network import Network
import collections
from itertools import groupby
from collections import Counter

from modules.lookup import unify_firm_name


def read_file(filename, input_path):
    input_path
    os.chdir(input_path)
    data = pd.read_excel(filename, engine="openpyxl", sheet_name="Sheet1")
    data = data[["WIPSONkey", "Assignee", "Gyeongnam", "IPC"]]
    return data 


def ThesFirm(data):
    # to add other stopwords >> , "stopword"
    stopwords = "(주)", "주식회사", "유", "유한회사", "산학협력단", "재단법인", "대한민국관리부서", "산학협력재단"
    stopwords_ = "|".join(map(re.escape, stopwords))
    data = re.sub(stopwords_, "", data)
    data = re.sub(r'[^\w\s]','', data)
    data2 = data.split(" ")
    unify_firm = unify_firm_name(data2)
    output = " ".join(unify_firm)
    return output



def func_bgram(data):
    stopwords = "(주)" # to add other stopwords >> , "stopword"
    stopwords_ = "|".join(map(re.escape, stopwords))
    data = re.sub(stopwords_, "", data)
    data = re.sub(r'[^\w\s]','', data)
    tokens = word_tokenize(data)
    bgram = bigrams(tokens)
    bgram_list = [x for x in bgram]
    return bgram_list


def empty_to_nan(data):
    if data == []:
        return np.nan
    else:
        return data

def split_column_into_lists(column):
    """Turn every rows of the column into list of lists"""
    column2 = column.copy()
    for i in range(len(column2)):
        try:
            column2.iloc[i] = str(column2.iloc[i]).split()
            for j in range(len(column2.iloc[i])):
                column2.iloc[i][j] = column2.iloc[i][j].strip()
        except:
            if isinstance(column2.iloc[i], str): # some columns consist of rows with list while others with string
                column2[i] = str(column2.iloc[i]).split()
                for j in range(len(column.iloc2[i])):
                    column2.iloc[i][j] = column2.iloc[i][j].strip()
            else:
                column2.iloc[i] = column2.iloc[i]
    column2list = column2
    return column2list


def column_to_list(data):
    """Make rows of a column into one list"""
    data2 = data
    list_ = []
    list_2 = split_column_into_lists(data2)
    for i in range(len(data2)):
        list_.extend(list_2.tolist()[i])
    # or 
    # firm_patent_list = [item for patent_list in t for item in firm_patent_list]
    return list_

def construct_NM_v2(data,column_, type_,):
    data[column_] = data[column_].swifter.apply(lambda x: func_bgram(x))
    column_series = data[column_].swifter.apply(lambda x: empty_to_nan(x)).dropna().reset_index(drop = True)
    # make a list of tuple
    column_list = []
    for i in range(len(column_series)):
        if len(column_series[i]) == 1:
            column_list.extend(column_series[i])
        else:
            for j in range(len(column_series[i])):
                column_list.append(column_series[i][j])
    # from list to dataframe
    network_df = pd.DataFrame(column_list, columns = ["firm1", "firm2"])
    network_df["index"] = network_df.index    
    # to weight the relationship        
    network_df2 = network_df.groupby(["firm1", "firm2"])["index"].count().reset_index(name = "count")
    return column_list, network_df2

def construct_NM(data, column_, type_): # type_ = 0 if total, 1 if limited scope
    data2 = data.copy()    
    if type_ == 0:
        column_list, network_df2 = construct_NM_v2(data2, column_, type_)        
    else: # to limit to the firms who have applied patents at least certain amount
        column_list_ = column_to_list(data2[column_])
        data3 = limit_scope(column_list_, data2, column_).reset_index()
        column_list, network_df2 = construct_NM_v2(data3, column_, type_)
    return column_list, network_df2

def basic_statistic(data, type_): # type_ : Assignee or IPC
    data2 = data.copy()    
    
    if type_ == "IPC": # in case of IPC, we need to choose only first four characters
        data2 = data2.loc[data2["Gyeongnam"] == True]
        # column to list
        data_in_list = column_to_list(data2[type_])
        data_in_list = [fourDigit(x) for x in data_in_list]
    else:
        # column to list
        data_in_list = column_to_list(data2[type_])
    firm_freq = list(Counter(data_in_list).values())
    firm_list = list(Counter(data_in_list).keys())
    freq_df = pd.DataFrame(zip(firm_list, firm_freq), columns = [type_, "freq"])
    return freq_df

def limit_scope(column_list_, data, column):
    # count records of each firm
    firm_freq = list(Counter(column_list_).values())
    firm_list = list(Counter(column_list_).keys())
    freq_list = pd.DataFrame(zip(firm_list, firm_freq), columns = ["firm", "freq"])
    data_at_least = freq_list.loc[freq_list["freq"] >= 10]
    data_at_least2 = data_at_least["firm"].tolist()
    
    data2 = data[pd.DataFrame(data[column].apply(lambda x: x.split()).tolist()).isin(data_at_least2).any(1).values]
    return data2

def edge_list(data, column_, type_): # type_ : 0 if analyzing all data // 1 if limiting the scope
    co_words_list = []    
    if type_ == 0:        
        data2 = data[column_].copy()
        data2 = data2.apply(lambda x: x.split(" "))
        data3 = data2
    else:
        data2 = data.copy()
        column_list_ = column_to_list(data2[column_])
        data3 = limit_scope(column_list_, data2, column_).reset_index()
        data3 = data3[column_]
        data3 = data3.apply(lambda x: x.split(" "))
    for i in range(len(data3)):        
        if len(data3[i]) == 1:
            co_words = [[], []]
            co_words[0] = "".join(data3[i])
            co_words[1] = np.nan
            co_words_list.append(co_words)
        else:
            # # remove duplicate Assignees.. DO NOT USE THIS when analyzing IPCs
            # data3[i] = list(set(data3[i]))
            # sort alphabetically to address non-directed relationship
            data3[i].sort()
            for j in range(len(data3[i])):
                for k in range(len(data3[i])):
                    if (j < k) & (j + k <= len(data3[i])*2 -1):
                        co_words = [[], []]
                        co_words[0] = data3[i][j]
                        co_words[1] = data3[i][k]
                        co_words_list.append(co_words)  
    co_words_df = pd.DataFrame(co_words_list, columns = ["firm1", "firm2"])
    return co_words_df


def fourDigit(x):
    if isinstance(x, str):
        return x[0:4]
    else:
        return x

def classify_group(data): # type_ == assignee or ipc
    data2 = data.copy()
    # conditions to classify groups
    large = ["대우조선해양", "삼성중공업", "한국조선해양", "현대중공업"]
    med = ["현대미포조선", "현대삼호중공업", "에이치제이중공업", "케이조선"]       
    # conditions = [data2[type_].isin(large), data2[type_].isin(med)]
    choices = ["대기업", "중견기업"]
    # data2["type"] = np.select(conditions, choices, default = "기타")
    return data2["type"]



def func_adjency(data):
    words = sorted(list(set(item for t in data for item in t )))
    df = pd.DataFrame(0, columns = words, index = words)
    for i in data:
        df.at[i[0],i[1]] +=1
    return df

# data = [("대우조선해양", "삼성중공업"), ("한국조선해양", "케이조선"), ("힝", "현대삼호중공업", "현대중공업")]


def plot_adjency(data):
    G = nx.from_pandas_adjacency(data)


    # G = nx.from_pandas_adjacency(df)
    large = ["대우조선해양", "삼성중공업", "한국조선해양", "현대중공업"]
    med = ["현대미포조선", "현대삼호중공업", "에이치제이중공업", "케이조선"]
    colors = []
    for node in G:
        if node in large:
            colors.append("blue")
        elif node in med:
            colors.append("green")
        else:
            colors.append("red")
    font_list = fm.findSystemFonts(fontpaths=None, fontext="ttf")
    font_location = [font_list[i] for i in range(len(font_list)) if "NanumGothic.ttf" in font_list[i]][0]
    font_name = fm.FontProperties(fname = font_location).get_name()
    plt.rc("font", family = font_name)
    nx.draw(G, font_family = font_name, with_labels = True, node_color = colors)
    plt.show()

# plt.matplotlib_fname()


# def tag_group(data):
#     data2 = data.copy()
#     data2 = pd.Series(data2, name = "firm").reset_index()
#     large = ["대우조선해양", "삼성중공업", "한국조선해양", "현대중공업"]
#     med = ["현대미포조선", "현대삼호중공업", "에이치제이중공업", "케이조선"]
#     conditions = [data2["firm"].isin(large), data2["firm"].isin(med)]
#     choices = ["대기업", "중견기업"]
#     data2["type"] = np.select(conditions, choices, default = "기타")
#     return data2["type"]


def match_assignee_ipc(firm, ipc):
    co_words_list = []
    for i in range(len(firm)):
        for j in range(len(ipc)):
            co_words = [[], []]
            co_words[0] = firm[i]
            co_words[1] = ipc[j][0:4]
            co_words_list.append(co_words)
    return co_words_list


def bipartite_edgeList(data):
    data2 = data.copy()
    co_words_list = []
    data2["Assignee"] = data2["Assignee"].apply(lambda x: x.split(" "))
    data2["IPC"] = data2["IPC"].apply(lambda x: x.split(" "))
    co_words_list = data2.apply(lambda x: match_assignee_ipc(x["Assignee"], x["IPC"]), axis = 1)    
    co_words_list2 = []
    for i in range(len(co_words_list)):
        for j in range(len(co_words_list[i])):
            co_words_list2.append(co_words_list[i][j])
    firm_ipc = pd.DataFrame(co_words_list2, columns = ["firm1", "firm2"]) # to simply use R code
    return firm_ipc


def multi_process(df, target_func, type_): # type_ = df // list // multipe_arg
    n_cores = 12
    # n_cores = 10
    if type_ ==  "df":        
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        df = pd.concat(pool.map(target_func, df_split))
    # elif type_ == "df2list":
    #     list_ = []
    #     df_split = np.array_split(df, n_cores)
    #     pool = Pool(n_cores)
    #     target_func2 = partial(target_func, "perf")
    #     df = list_.append(pool.map(target_func2, df_split))
    elif type_ == "list":
        list_ = []        
        list_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        df = list_.append(pool.map(target_func, list_split))
    elif type_ == "multiple_arg": # https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        # target_func_ = partial(target_func, multi_)
        df = pd.concat(pool.map(target_func, df_split))
    """
    pool.apply: the function call is performed in a seperate process / blocks until the function is completed / lack of reducing time
    pool.apply_async: returns immediately instead of waiting for the result / the orders are not the same as the order of the calls
    pool.map: list of jobs in one time (concurrence) / block / ordered-results
    pool.map_async: 
    http://blog.shenwei.me/python-multiprocessing-pool-difference-between-map-apply-map_async-apply_async/
    """
    pool.close()
    pool.join()
    return df