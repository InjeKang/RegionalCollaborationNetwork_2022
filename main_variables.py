from doctest import OutputChecker
from modules.GlobalVariables import *
from modules.AlliancePerformance import *

from functools import partial
from os.path import join
import multiprocessing as mp
import pandas as pd
import swifter

input_path = join(os.getcwd(), "data")

def main():
    data = functions_(
        # read and clean data
        read_data = True,
        clean_data = True,
        # constructing a network matrix
        descriptive = False,
        network_matrix = False,
        bipartite = True)

def functions_(read_data, clean_data, descriptive, network_matrix, bipartite):
    #  for v2, removed whitespace and then "\" is removed in v2
    raw_data = func_read_data("00.patent_total_v3(Thesa).xlsx", 0, read_data)    
    clean_output = func_clean_data(raw_data, clean_data)
    output_descriptive = func_descriptive(raw_data, "IPC", descriptive)
    network_df = func_network_matrix(raw_data, "IPC", network_matrix)
    bipartite_df = func_bipartite(clean_output, bipartite)
    # network_plot = func_network_visualize(network_list, network_df, network_visualize)
    return raw_data

def func_read_data(file_name, type_, read_data): # type_ : 0 if overall, 1 if certain firm/ipc selected
    if read_data:
        default_directory = os.getcwd()
        input_path = join(default_directory, "data")
        raw_data = read_file(file_name, input_path)
        if type_ == 1:
            """
            대기업: 삼성중공업, 대우조선해양, 한국조선해양, 현대중공업
            중견기업: 현대미포조선, 현대삼호중공업, 에이치제이중공업, 케이조선
            IPC: B60V, B63B, B63C, B63G, B63H, B63J
            현대계열: 한국조선해양, 현대중공업, 현대미포조선, 현대삼호중공업
            """
            raw_data = raw_data.loc[
                            (raw_data["Assignee"].str.contains("현대미포조선")) |
                            (raw_data["Assignee"].str.contains("현대삼호중공업")) |
                            (raw_data["Assignee"].str.contains("현대중공업")) |
                            (raw_data["Assignee"].str.contains("한국조선해양"))].reset_index()
            # raw_data = raw_data.loc[raw_data["Assignee"].str.contains("한국조선해양기자재연구원") == False].reset_index()

        return raw_data

def func_clean_data(data, clean_data):
    if clean_data:
        data["Assignee"] = data["Assignee"].swifter.apply(lambda x: ThesFirm(x))
        data.to_excel("00.patent_total_v4(Thesa2).xlsx", index = False)
        return data

def func_descriptive(data, column_, descriptive): # column_ : Assignee or IPC 
    if descriptive:
        if column_ == "Assigne":
            data2 = data.copy()
            # descriptive_total
            output = basic_statistic(data2, column_)
            output.to_excel("00.descriptive_summary_firm_total.xlsx", index = False)        
            # descriptive_network: single-assigned patents are excluded
            data3 = data2.loc[data2[column_].str.contains(" ")].reset_index(drop=True, inplace=False)
            output2 = basic_statistic(data3, column_)
            output2.to_excel("00.descriptive_summary_firm_network.xlsx", index = False)
        else: # column_ == "IPC"
            data2 = data.loc[data["Gyeongnam"] == True].reset_index(drop=True, inplace=False)
            # descriptive_total
            output = basic_statistic(data2, column_)
            output.to_excel("00.descriptive_summary_IPCB63B_total.xlsx", index = False)        
            # descriptive_network: single-assigned patents are excluded
            data3 = data2.loc[data2[column_].str.contains(" ")].reset_index(drop=True, inplace=False)
            output2 = basic_statistic(data3, column_)
            output2.to_excel("00.descriptive_summary_IPCB63B_network.xlsx", index = False)
        return output2

def func_network_matrix(data, column_, network_matrix):
    if network_matrix:        
        if column_ == "Assignee":
            data2 = data.loc[data[column_].str.contains(" ")].reset_index(drop=True, inplace=False)
            output = edge_list(data2, column_, 0) # type_ : 0 if analyzing all data // 1 if limiting the scope
            output.to_excel("01.network_matrix_df_v5(MedFirm).xlsx", index = False)
        else: # column_ == "IPC"
            data2 = data.loc[(data["Gyeongnam"] == True) &
                                (data[column_].str.contains(" "))].reset_index(drop=True, inplace=False)
            output = edge_list(data2, column_, 0) # in this function do not use list(set(data3[i]))
            output["firm1"] = output["firm1"].apply(lambda x: x[0:4])
            output["firm2"] = output["firm2"].apply(fourDigit)
            output.to_excel("01.network_matrix_df_v7(B60V).xlsx", index = False)
        # data2 = classify_group(data2) # firm or ipc
        # data3 = func_adjency(column_list)
        
        # data3.to_excel("network_matrix_adj.xlsx")
        return output

def func_bipartite(data, bipartite):
    if bipartite:
        # firm_list = ["삼성중공업", "대우조선해양", "한국조선해양", "현대중공업",
        #                 "현대미포조선", "현대삼호중공업", "에이치제이중공업", "케이조선"]
        firm_list = ["한국조선해양", "현대중공업", "현대미포조선", "현대삼호중공업"]
        ipc_list = ["B63J", "B63H", "F17C", "C02F", "E02B", "B63J", "F16L", "B63G", "G06Q", "G01N", "B63B", "G01S", "G08B", "B66F", "G01D", "B67D", "G05D", "B32B", "G08C", "H01M", "B63C", "F41F", "G01M", "F25J", "G06T", "A62D", "F02M", "F01N", "B23K", "F16M", "B01D", "B04B", "B64C", "B66C", "E01H", "F04D", "B29C", "B29K"]
        # ipc_list = ["B63B", "B63J", "G06Q", "G06F", "B63H", "F17C", "A47B", "B23K", "B63B", "F17C", "B63C", "G01M", "G06F", "B63B", "B63H", "B23K", "E02F", "F24F", "B63B", "F17C", "B63H", "B65D", "F02M"]
        # data2 = data.loc[(data["Assignee"].str.contains(" "))].reset_index(drop=True, inplace=False)
        data3 = data.loc[(pd.DataFrame(data["Assignee"].apply(lambda x: 
                        x.split()).tolist())).isin(firm_list).any(1).values].reset_index(drop=True, inplace=False)        
        # output = multi_process(data3, bipartite_edgeList, "df")
        output = bipartite_edgeList(data3)
        # only select data of firms
        output2 = output.loc[(pd.DataFrame(output["firm1"])).isin(firm_list).any(1).values].reset_index(drop=True, inplace=False)
        output2 = output2.loc[(pd.DataFrame(output2["firm2"])).isin(ipc_list).any(1).values].reset_index(drop=True, inplace=False)
        output2.to_excel("03.bipartite_edgeListHyundai.xlsx", index = False)             
        return output2



if __name__ == "__main__":
    main()
