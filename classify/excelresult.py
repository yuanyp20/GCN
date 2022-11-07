import pandas as pd
import sys
csv_file = sys.argv[1]
def excel_result_csv(csv_file):
    print(csv_file)
    csv = pd.read_csv(csv_file,encoding='utf-8',index_col=None,header=None)
    csv.info()
    csv.to_excel(csv_file[:-3]+'xlsx',encoding='utf-8')
excel_result_csv(csv_file)
