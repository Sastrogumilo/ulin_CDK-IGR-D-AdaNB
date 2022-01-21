import pandas as pd

header = ['age','bp','sg','al','su','rbc','pc','pcc',
    'ba','bgr','bu','sc','sod','pot','hemo','pcv',
    'wbcc','rbcc','htn','dm','cad','appet','pe','ane',
    'classification']

def arff_convert(filename):
    df = pd.read_csv("./media/"+filename, header=None, names=header)
    df = df.dropna(axis=0, how='any')
    df = df.rename({'wbcc':'wc', 'rbcc':'rc'}, axis='columns')
    df.to_csv('./media/arff.csv', index=False)
    df2 = pd.read_csv("./media/arff.csv")
    
    df2['id'] = df2.index + 1
    print(df2)
    df2 = df2[['id', 'age','bp','sg','al','su','rbc','pc','pcc',
    'ba','bgr','bu','sc','sod','pot','hemo','pcv',
    'wc','rc','htn','dm','cad','appet','pe','ane',
    'classification']]
    
    df2.to_csv('./media/dataset.csv', index=False)
    print('Sudah...')