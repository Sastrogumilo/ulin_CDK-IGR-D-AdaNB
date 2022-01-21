import pandas as pd
from django.core.files.storage import default_storage

def datatableConverter(data):
    
    dataset = []
    #data = pd.read_csv(default_storage.path('dataset.csv'))
        
    for x in range(len(data['age'])):
        temp = []
        temp.append(data['id'][x])
        temp.append(data['age'][x])
        temp.append(data['bp'][x])
        temp.append(data['sg'][x])
        temp.append(data['al'][x])
        temp.append(data['su'][x])
        temp.append(data['rbc'][x])
        temp.append(data['pc'][x])
        temp.append(data['pcc'][x])
        temp.append(data['ba'][x])
        temp.append(data['bgr'][x])
        temp.append(data['bu'][x])
        temp.append(data['sc'][x])
        temp.append(data['sod'][x])
        temp.append(data['pot'][x])
        temp.append(data['hemo'][x])
        temp.append(data['pcv'][x])
        temp.append(data['wc'][x])
        temp.append(data['rc'][x])
        temp.append(data['htn'][x])
        temp.append(data['dm'][x])
        temp.append(data['cad'][x])
        temp.append(data['appet'][x])
        temp.append(data['pe'][x])
        temp.append(data['ane'][x])
        temp.append(data['classification'][x])
        
        # print(data['sentiment'][x])
        dataset.append(temp)
    return dataset

def datatableConverter2(data):
    
    
    data['id'] = data.index + 1
    print(data)
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    
    
    data_fitur = []
    #data = pd.read_csv(default_storage.path('dataset_encode.csv'))
    for x in range(len(data['age'])):
        temp = []
        temp.append(data['id'][x])
        temp.append(data['age'][x])
        temp.append(data['blood_pressure'][x])
        temp.append(data['specific_gravity'][x])
        temp.append(data['albumin'][x])
        temp.append(data['sugar'][x])
        temp.append(data['red_blood_cells'][x])
        temp.append(data['pus_cell'][x])
        temp.append(data['pus_cell_clumps'][x])
        temp.append(data['bacteria'][x])
        temp.append(data['blood_glucose_random'][x])
        temp.append(data['blood_urea'][x])
        temp.append(data['serum_creatinine'][x])
        temp.append(data['sodium'][x])
        temp.append(data['potassium'][x])
        temp.append(data['haemoglobin'][x])
        temp.append(data['packed_cell_volume'][x])
        temp.append(data['white_blood_cell_count'][x])
        temp.append(data['red_blood_cell_count'][x])
        temp.append(data['hypertension'][x])
        temp.append(data['diabetes_mellitus'][x])
        temp.append(data['coronary_artery_disease'][x])
        temp.append(data['appetite'][x])
        temp.append(data['pedal_edema'][x])
        temp.append(data['anemia'][x])
        temp.append(data['classification'][x])
        
        # print(data['sentiment'][x])
        data_fitur.append(temp)
    return data_fitur