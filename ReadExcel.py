import numpy as np
import matplotlib.pyplot as plt

# This function is used in the function readExcel(...) defined further below
def readExcelSheet1(excelfile):
    from pandas import read_excel
    return (read_excel(excelfile)).values

# This function is used in the function readExcel(...) defined further below
def readExcelRange(excelfile,sheetname="Sheet1",startrow=1,endrow=1,startcol=1,endcol=1):
    from pandas import read_excel
    values=(read_excel(excelfile, sheetname,header=None)).values
    return values[startrow-1:endrow,startcol-1:endcol]

# This is the function you can actually use within your program.
# See manner of usage further below in the section "Prepare Data"
def readExcel(excelfile,**args):
    if args:
        data=readExcelRange(excelfile,**args)
    else:
        data=readExcelSheet1(excelfile)
    if data.shape==(1,1):
        return data[0,0]
    elif (data.shape)[0]==1:
        return data[0]
    else:
        return data

# This function get the names of the sheets in the excel file
def getSheetNames(excelfile):
    from pandas import ExcelFile
    return (ExcelFile(excelfile)).sheet_names

