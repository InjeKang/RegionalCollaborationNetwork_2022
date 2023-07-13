from numpy import average
import pandas as pd
import os
from os.path import join
from statistics import mean
import numpy as np

os.chdir(join(os.getcwd(), "data"))
data = pd.read_excel("00.patent_total_v4(Thesa2).xlsx", engine="openpyxl", sheet_name="Sheet1")
data = data[["weight", "from_label", "to_label"]]

co_words_list = []
y = [["세보테크놀로지"], ["한국조선해양기자재연구원", "세보테크놀로지"], ["세보테크놀로지"], ["한국조선해양기자재연구원", "세보테크놀로지"]]
x = pd.Series(i for i in y)
for i in range(len(x)):        
    if len(x[i]) == 1:
        co_words = [[], []]
        co_words[0] = "".join(x[i])
        co_words[1] = np.nan
        co_words_list.append(co_words)
        print(co_words)
    else:
        x[i].sort()
        for j in range(len(x[i])):
            for k in range(len(x[i])):
                if (j < k) & (j + k <= len(x[i])*2-1):
                    co_words = [[], []]
                    co_words[0] = x[i][j]
                    co_words[1] = x[i][k]
                    co_words_list.append(co_words)
                    print(co_words)
co_words_list

# B60V, B63B, B63C, B63G, B63H, B63J

data2 = data.loc[(data["from_label"].str.contains("대우조선해양")) | data["to_label"].str.contains("대우조선해양")]

data2 = data2.sort_values(["weight"], ascending=[False])
data2.head()

test1 = raw_data[raw_data["IPC"].str.contains("B60V")]
test2 = raw_data[raw_data["IPC"].str.contains("B63B")]
test3 = raw_data[raw_data["IPC"].str.contains("B63C")]
test4 = raw_data[raw_data["IPC"].str.contains("B63G")]
test5 = raw_data[raw_data["IPC"].str.contains("B63H")]
test6 = raw_data[raw_data["IPC"].str.contains("B63J")]

raw_data = raw_data[(raw_data["Gyeongnam"] == True) & (raw_data["IPC"].str.contains(" "))]
len(test1)
len(test2)



test = data.loc[(data["Assignee"].str.contains("한국조선해양")) & 
                        (data["Assignee"].str.contains(" "))].reset_index(drop = True)
test3 = test["Assignee"].apply(lambda x : x.split(" "))
no = []
for i in range(len(test3)):
    no.append(len(test3[i]))
print(mean(no))
len(test3)



data_in_list = column_to_list(raw_data["IPC"])
data_in_list = [fourDigit(x) for x in data_in_list]
firm_freq = list(Counter(data_in_list).values())
firm_list = list(Counter(data_in_list).keys())
freq_df = pd.DataFrame(zip(firm_list, firm_freq), columns = ["IPC", "freq"])

freq_df.sort_values("freq", ascending = False)

ipc_list = ["B63B", "B63J", "G06Q", "G06F", "B63H", "F17C", "A47B", "B23K", "B63B", "F17C", "B63C", "G01M", "G06F", "B63B", "B63H", "B23K", "E02F", "F24F", "B63B", "F17C", "B63H", "B65D", "F02M"]
ipc_list2 = list(set(ipc_list))
ipc_list2.sort()
ipc_list2
