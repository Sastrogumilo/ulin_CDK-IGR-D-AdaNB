import pandas as pd

header = ['age','bp','sg','al','su','rbc','pc','pcc',
    'ba','bgr','bu','sc','sod','pot','hemo','pcv',
    'wbcc','rbcc','htn','dm','cad','appet','pe','ane',
    'classification']

def arff_convert(filename):
    df = pd.read_csv("./media/"+filename, header=None, names=header)
    df = df.dropna(axis=0, how='any')
    df = df.rename({'wbcc':'wc', 'rbcc':'rc'}, axis='columns')
    df['id'] = df.index + 1
    print(df)
    df.to_csv('./media/dataset.csv', index=False)
    print('Sudah...')