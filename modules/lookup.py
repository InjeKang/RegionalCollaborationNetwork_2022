import pandas as pd
import numpy as np
import copy

def create_placeholder_list(list_of_list):
    """Create a placeholder list (a) to prevent error of index out of range and
    (b) to have a word (instead of an alphabet) as a string"""
    if isinstance(list_of_list, list):
        placeholder_list = []
        for i, _ in enumerate(list_of_list):
            placeholder_list_row = []
            for j, _ in enumerate(list_of_list[i]):
                placeholder_list_row.append("")
            placeholder_list.append(placeholder_list_row)
        return placeholder_list
    else:
        placeholder_list = []
        for i, _ in enumerate(list_of_list):
            placeholder_list_row = []
            for j, _ in enumerate(list_of_list.iloc[i]):
                placeholder_list_row.append("")
            placeholder_list.append(placeholder_list_row)
        return placeholder_list


def unify_firm_name(data): 
    """making dictionary to deal with the name of institutes with different expression"""    
    # make a lookup table
    lookup_table = pd.DataFrame(np.array([
        ["한국해양연구원", "한국해양과학기술원"],
        ["대우조선해양이엔알", "대우조선해양"],
        ["현대중공업터보기계", "현대중공업"],
        ["한국조선해양기자재연구원", "한조해기자재연구원"], # to differentiate with "한국조선해양"

        ["대한민국해양경찰청해양경찰연구센터장", "해양경찰청"],
        ["대한민국해양경찰청장", "해양경찰청"]
        
        ]), columns = ["before", "after"])

    institute = data.copy()    
    institute_matched = [""]*len(institute)
    for i, _ in enumerate(institute): # if for function not used, empty results...don't know why
        for name, institute_ in zip(lookup_table["before"], lookup_table["after"]):
            if _ == name:
                institute_matched[i] = institute_
        if institute_matched[i] == "":             
            institute_matched[i] = "".join(institute[i])
    institute_matched2 = institute_matched
    return institute_matched2 # in list type
