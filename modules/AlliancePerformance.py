import multiprocessing as mp
import pandas as pd
import os
import openpyxl
import glob
from tqdm import trange
from multiprocessing import Pool, cpu_count
import numpy as np
from os.path import join
from modules.GlobalVariables import *
from modules.lookup import *


def performance_stock(patent_data, data):
    """
    accumulating patent data applied after the alliance 
    """
    time_ = 10
    alliance_data = data.copy()
    # patent_data = multi_process(excel_files, read_xlsx_glob, "list")
    # alliance_data["performance_t10"] = alliance_data.apply(lambda x: alliance_with_patent(x, patent_data, time_), axis = 1)
    alliance_data["performance_t10"] = alliance_data.swifter.apply(lambda x: alliance_with_patent(
        x["focal"], x["partner"], x["year"], patent_data, time_), axis = 1)
    print(len(alliance_data))
    return alliance_data

def alliance_with_patent(focal, partner, year_, patent_data, time_):
    """
    retrieving patents (1) assigned by both focal/partner firms (2) after t yeas of alliance
    """    
    patent = patent_data.copy()
    patent = patent.reset_index(drop=True, inplace=False)
    # retrieving patents assigned during the designated period
    year = int(year_)
    patent2 = patent[(pd.to_numeric(patent[10]) <= year + time_)].reset_index(drop=True, inplace=False)
    # removing duplicated data
    patent2 = patent2.drop_duplicates(subset = [9]).reset_index(drop=True, inplace=False)
    # retrieving patent assigned by both firms
    patent3 = pd.Series(affil_firm(patent2, "perf")) # the function is from StrategicDiagram_npi
    patent4 = patent3[(patent3.str.contains(focal.lower())) &
    (patent3.str.contains(partner.lower()))]

    # number of performance data
    output = len(patent4)
    return output