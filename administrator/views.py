from django.core.files import storage
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.utils.decorators import decorator_from_middleware
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib import messages
import random
from operator import itemgetter
import zipfile
import os
from shutil import copyfile

#Plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import plot
from plotly.subplots import make_subplots

#End Plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from time import time

#Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn import naive_bayes
from sklearn.metrics import *
from sklearn.ensemble import AdaBoostClassifier


import time
import numpy as np

from .arff_convert import arff_convert


header = ['age','bp','sg','al','su','rbc','pc','pcc',
    'ba','bgr','bu','sc','sod','pot','hemo','pcv',
    'wbcc','rbcc','htn','dm','cad','appet','pe','ane',
    'classification']


# Create your views here.

@login_required(login_url=settings.LOGIN_URL)
def index(request):
    return render(request, 'administrator/dashboard.html')

@login_required(login_url=settings.LOGIN_URL)
def tentang(request):
    return render(request, 'administrator/tentang.html')

#@login_required(login_url=settings.LOGIN_URL)
#def SVM(request):
#    return render(request, 'administrator/SVM.html')

@login_required(login_url=settings.LOGIN_URL)
def NaiveBayes_Algen(request):
    return render(request, 'administrator/NaiveBayes_Algen.html')

#@login_required(login_url=settings.LOGIN_URL)
#def SVMRBFIG(request):
#    return render(request, 'administrator/SVMRBFIG.html')


##Main Module


@login_required(login_url=settings.LOGIN_URL)
def dataset(request):
    
    if request.method == 'POST':
        file = request.FILES['data']
        
        if file:
            #if default_storage.exists(file):
            #    default_storage.delete(file)
            filename = file.name
            
            print(filename)
            if filename.endswith('.arff'):
                if not default_storage.exists(file.name):
                    default_storage.save(file.name, file)
                
                #else:
                  #  default_storage.delete(file)
                
                    
        if default_storage.exists('dataset.csv'):
            default_storage.delete('dataset.csv')
            file_name = default_storage.save('dataset.csv', file)

        if filename.endswith('.arff'):
            arff_convert(filename)

        dataset = []
        data = pd.read_csv(default_storage.path('dataset.csv'))
        
        for x in range(len(data['age'])):
            temp = []
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
        # path = default_storage.save('dataset.csv', ContentFile(file.read()))
        messages.success(request,'Dataset berhasil diupload!')

        return render(request, 'administrator/dataset.html',{'dataset': dataset})
    else:
        if default_storage.exists('dataset.csv'):
            dataset = []
            data = pd.read_csv(default_storage.path('dataset.csv'))
       
        for x in range(len(data['age'])):
            temp = []
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

            # with open(default_storage.path('dataset.csv'), 'r') as data:
            #     reader = csv.reader(data)
            #     dataset = []
            #     for row in reader:
            #         dataset.append(row)
            # print(dataset)

        else:
            dataset = []
        # nama=[]
        # jumlah=[]
        # dataset=[]
        # if default_storage.exists('dataset'):
        #     for name in os.listdir(os.path.join(settings.BASE_DIR, 'media/dataset')):
        #         dataset.append([str(name),str(len(os.listdir(os.path.join(settings.BASE_DIR, 'media/dataset/'+name))))])
        # # print(dataset)
        
        return render(request, 'administrator/dataset.html',{'dataset': dataset})



@login_required(login_url=settings.LOGIN_URL)
def EDA(request):
    
    if default_storage.exists('dataset_encode.csv'):
        data = pd.read_csv(default_storage.path('dataset_encode.csv'))
        
        #Graf Kelas
        """
        kelas_data = px.bar(
                            data, x = data['Class'].unique(), y = data['Class'].value_counts(), 
                            color = data['Class'].unique(), 
                            title="Graf Kelas",
                            labels={
                                "x": "Kelas",
                                "y": "Jumlah"
                                }
                            )
        plot_kelas = plot(kelas_data, output_type='div')
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
            #temp.append(data['classification'][x])
        """
        name_plot = ["Age", "Blood Presure", "Specific Gravity", "Albumin", "Sugar",
                     "Red Blood Cells", "Pus Cell", "Pus Cell Clumps", "Bacteria",
                     "Blood Glucose Random", "Blood Urea", "Serum Creatinine",
                     "Sodium", "Potassium", "Haemoglobin", "Packed Cell Volume",
                     "White Blood Cell Count", "Red Blood Cell Count", "Hypertension",
                     "Diabetes Mellitus", "Coronary Artery Disease", "Appetite",
                     "Pedal Edema", "Anemia"
                     ]
        #print(data[0])
        plot_data_utama = make_subplots(rows=4, cols=6, subplot_titles=name_plot,) 
        plot_data_utama.add_trace(go.Histogram(x=data['age'], name=name_plot[0]), col=1, row=1)
        plot_data_utama.add_trace(go.Histogram(x=data['blood_pressure'], name=name_plot[1]), col=2, row=1)
        plot_data_utama.add_trace(go.Histogram(x=data['specific_gravity'], name=name_plot[2]), col=3, row=1)
        plot_data_utama.add_trace(go.Histogram(x=data['albumin'], name=name_plot[3]), col=4, row=1)
        plot_data_utama.add_trace(go.Histogram(x=data['sugar'], name=name_plot[4]), col=5, row=1)
        plot_data_utama.add_trace(go.Histogram(x=data['red_blood_cells'], name=name_plot[5]), col=6, row=1)
        plot_data_utama.add_trace(go.Histogram(x=data['pus_cell'], name=name_plot[6]), col=1, row=2)
        plot_data_utama.add_trace(go.Histogram(x=data['pus_cell_clumps'], name=name_plot[7]), col=2, row=2)
        plot_data_utama.add_trace(go.Histogram(x=data['bacteria'], name=name_plot[8]), col=3, row=2)
        plot_data_utama.add_trace(go.Histogram(x=data['blood_glucose_random'], name=name_plot[9]), col=4, row=2)
        plot_data_utama.add_trace(go.Histogram(x=data['blood_urea'], name=name_plot[10]), col=5, row=2)
        plot_data_utama.add_trace(go.Histogram(x=data['serum_creatinine'], name=name_plot[11]), col=6, row=2)
        plot_data_utama.add_trace(go.Histogram(x=data['sodium'], name=name_plot[12]), col=1, row=3)
        plot_data_utama.add_trace(go.Histogram(x=data['potassium'], name=name_plot[13]), col=2, row=3)
        plot_data_utama.add_trace(go.Histogram(x=data['haemoglobin'], name=name_plot[14]), col=3, row=3)
        plot_data_utama.add_trace(go.Histogram(x=data['packed_cell_volume'], name=name_plot[15]), col=4, row=3)
        plot_data_utama.add_trace(go.Histogram(x=data['white_blood_cell_count'], name=name_plot[16]), col=5, row=3)
        plot_data_utama.add_trace(go.Histogram(x=data['red_blood_cell_count'], name=name_plot[17]), col=6, row=3)
        plot_data_utama.add_trace(go.Histogram(x=data['hypertension'], name=name_plot[18]), col=1, row=4)
        plot_data_utama.add_trace(go.Histogram(x=data['diabetes_mellitus'], name=name_plot[19]), col=2, row=4)
        plot_data_utama.add_trace(go.Histogram(x=data['coronary_artery_disease'], name=name_plot[20]), col=3, row=4)
        plot_data_utama.add_trace(go.Histogram(x=data['appetite'], name=name_plot[21]), col=4, row=4)
        plot_data_utama.add_trace(go.Histogram(x=data['pedal_edema'], name=name_plot[22]), col=5, row=4)
        plot_data_utama.add_trace(go.Histogram(x=data['anemia'], name=name_plot[23]), col=6, row=4)
        
        #FIG Update
        plot_data_utama.update_layout(height=800, width=1248, title_text="Info Label", title_font_size=20, title_x=0.5)
        
        #Fig Export
        plot_data_div = plot(plot_data_utama, output_type='div')
        
        
        #Correlation Matrix
        data2 = data.drop('classification', axis=1)
        dfc = data2.corr()
        z = dfc.values.tolist()
        z_text = [[str(round(y,1)) for y in x] for x in z]
        
        plot_correlation = ff.create_annotated_heatmap(
                z=z,
                x=list(data2.columns),
                y=list(data2.columns),
                zmax=1, zmin=-1,
                annotation_text=z_text, colorscale='agsunset',
                showscale=True,
                hoverongaps=True
            )

        plot_correlation.update_layout(title_text='Correlation matrix', height=800, width=1096, title_x=0.5, title_y=1, title_font_size=20)

        plot_correlation_div = plot(plot_correlation, output_type='div')
        
        
        #Distribution Diagram 
        
        data2 = pd.read_csv('./media/dataset.csv')
        
        #net_category_dm = data['diabetes_mellitus'].value_counts().to_frame().reset_index().rename(columns={'index':'diabetes_mellitus','diabetes_mellitus':'count'})
        #net_category_htn=data['hypertension'].value_counts().to_frame().reset_index().rename(columns={'index':'hypertension','hypertension':'count'})
        #net_category_pcc=data['pus_cell_clumps'].value_counts().to_frame().reset_index().rename(columns={'index':'pus_cell_clumps','pus_cell_clumps':'count'})
        #net_category_ane=data['anemia'].value_counts().to_frame().reset_index().rename(columns={'index':'anemia','anemia':'count'})
        net_category_class = data['classification'].value_counts().to_frame().reset_index().rename(columns={'index':'classification','classification':'count'})
        
        colors=['orange','lightskyblue']
        
        dist_diagram = make_subplots(rows=1, cols=1, subplot_titles=["Classification",],
                                     specs=[[{"type": "pie"}, ],])
        
        dist_diagram.add_trace(go.Pie(
                                labels=net_category_class['classification'], 
                                values=net_category_class['count'],
                                name="Classifications"), 
                                col=1, row=1)
        
        #dist_diagram.add_trace(go.Pie(
        #                        labels=net_category_htn['hypertension'], 
        #                        values=net_category_htn['count'],
        #                        name='Hypertension'), 
        #                       col=2, row=1)
        

 
        #dist_dm = go.Figure([go.Pie(labels=net_category_dm['diabetes_mellitus'], values=net_category_dm['count'])])

        dist_diagram.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))
        
        dist_diagram.update_layout(title="Diagram Distribution",title_x=0.5)
        
        dist_diagram_div = plot(dist_diagram, output_type='div')
            
        
        #dist_diagram_2 = make_subplots(rows=1, cols=2, subplot_titles=["Pus Cell Clumps", "Anemia"],
        #                             specs=[[{"type": "pie"}, {"type": "pie"}],])
        
        #dist_diagram_2.add_trace(go.Pie(
        #                        labels=net_category_pcc['pus_cell_clumps'], 
        #                        values=net_category_pcc['count'],
        #                        name='Pus Cell Clumps'),
        #                        col=1, row=1)
        
        #dist_diagram_2.add_trace(go.Pie(
        #                        labels=net_category_ane['anemia'], 
        #                        values=net_category_ane['count'],
        #                        name='Anemia'),
        #                        col=2, row=1)
        
        #dist_diagram_2.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))
        
        #dist_diagram_2_div = plot(dist_diagram_2, output_type='div')
        
        return render(request, 'administrator/EDA.html',context={
                                                                'plot_div_kelas': plot_data_div,
                                                                'plot_div_korelasi': plot_correlation_div,
                                                                'plot_div_diagram': dist_diagram_div,
                                                                #'plot_div_diagram_2': dist_diagram_2_div,
                                                                
                                                                                                                        
                                                                }
                    
                    )
    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass


@login_required(login_url=settings.LOGIN_URL)

def preproses(request):
    if default_storage.exists('dataset.csv'):
        
        CKD_dataset = pd.read_csv("./media/dataset.csv", header=0, na_values="?")
        cols_names = {"bp": "blood_pressure",
                    "sg": "specific_gravity",
                    "al": "albumin",
                    "su": "sugar",
                    "rbc": "red_blood_cells",
                    "pc": "pus_cell",
                    "pcc": "pus_cell_clumps",
                    "ba": "bacteria",
                    "bgr": "blood_glucose_random",
                    "bu": "blood_urea",
                    "sc": "serum_creatinine",
                    "sod": "sodium",
                    "pot": "potassium",
                    "hemo": "haemoglobin",
                    "pcv": "packed_cell_volume",
                    "wc": "white_blood_cell_count",
                    "rc": "red_blood_cell_count",
                    "htn": "hypertension",
                    "dm": "diabetes_mellitus",
                    "cad": "coronary_artery_disease",
                    "appet": "appetite",
                    "pe": "pedal_edema",
                    "ane": "anemia"}

        CKD_dataset.rename(columns=cols_names, inplace=True)
        print(f"\nSudah ...")

        # Change to Numerical Dtyp
        CKD_dataset['red_blood_cell_count'] = pd.to_numeric(CKD_dataset['red_blood_cell_count'], errors='coerce')
        CKD_dataset['packed_cell_volume'] = pd.to_numeric(CKD_dataset['packed_cell_volume'], errors='coerce')
        CKD_dataset['white_blood_cell_count'] = pd.to_numeric(CKD_dataset['white_blood_cell_count'], errors='coerce')

        # Drop id Column as it is seems to be an unique identifier for each row
        CKD_dataset.drop(["id"], axis=1, inplace=True)

        # Checking missing values
        CKD_dataset.isnull().sum().sort_values(ascending=False)

        # Replace incorrect values
        CKD_dataset['diabetes_mellitus'] = CKD_dataset['diabetes_mellitus'].replace(
            to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'})
        CKD_dataset['coronary_artery_disease'] = CKD_dataset['coronary_artery_disease'].replace(to_replace='\tno', value='no')
        CKD_dataset['classification'] = CKD_dataset['classification'].replace(to_replace='ckd\t', value='ckd')

        # Convert nominal values to binary values
        CKD_dataset.replace("?", np.NaN, inplace=True)
        conv_value = {"red_blood_cells": {"normal": 1, "abnormal": 0},
                    "pus_cell": {"normal": 1, "abnormal": 0},
                    "pus_cell_clumps": {"present": 1, "notpresent": 0},
                    "bacteria": {"present": 1, "notpresent": 0},
                    "hypertension": {"yes": 1, "no": 0},
                    "diabetes_mellitus": {"yes": 1, "no": 0},
                    "coronary_artery_disease": {"yes": 1, "no": 0},
                    "appetite": {"good": 1, "poor": 0},
                    "pedal_edema": {"yes": 1, "no": 0},
                    "anemia": {"yes": 1, "no": 0},
                    "classification": {"ckd": 1, "notckd": 0}}
        CKD_dataset.replace(conv_value, inplace=True)

        # Fill null values with mean value of the respective column
        CKD_dataset.fillna(round(CKD_dataset.mean(), 2), inplace=True)

        # Save the final data cleaning
        CKD_dataset.to_csv("./media/dataset_encode.csv", sep=',', index=False)
        
        data_fitur = []
        data = pd.read_csv(default_storage.path('dataset_encode.csv'))
        for x in range(len(data['age'])):
            temp = []
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
            #temp.append(data['classification'][x])
            
            # print(data['sentiment'][x])
            data_fitur.append(temp)
            
        data_target = []
        target = pd.read_csv(default_storage.path('dataset_encode.csv'))
        for x in range(len(target['classification'])):
            temp = []
            temp.append(target['classification'][x])
            data_target.append(temp)
            
        return render(request, 'administrator/preproses.html',{'DataFitur': data_fitur, 'DataTarget': data_target})
    else:
        messages.error(request, 'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass

@login_required(login_url=settings.LOGIN_URL)
def NaiveBayes(request):
    if default_storage.exists('dataset_encode.csv'):
        #Target = pd.read_csv(default_storage.path('target.csv'))
        dataset = pd.read_csv('./media/dataset_encode.csv')
        X = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].values
        sc = StandardScaler()
        X = sc.fit_transform(X)
        
        X_train , X_test , y_train , y_test   = train_test_split(X,y,test_size = 0.2 , random_state=420, shuffle=True)
        
        #print(X_test)
        #print(y_test)
        #print(X_train.columns)
        
        Logit_Model = naive_bayes.GaussianNB()
        Logit_Model.fit(X_train, y_train)
        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=69696969)
        n_scores_1 = cross_val_score(Logit_Model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        Train = n_scores_1.mean()
        
        Prediction = Logit_Model.predict(X_test)
        Val_Score = accuracy_score(y_test,Prediction)
        Report = classification_report(y_test,Prediction)

        print(Prediction)
        print("Accuracy Score: {}%".format(Val_Score*100))
        print(Report)
        
        labels=['NonCKD', 'CKD']
        preds = np.array(Logit_Model.predict(X_test))
        #preds2 = Logit_Model.score(X_test, y_test)
        #preds = np.argmax(preds, axis = -1)
        orig = y_test
        conf = confusion_matrix(orig, preds)
        
        fig = ff.create_annotated_heatmap(conf, colorscale='blues', x=labels, y=labels)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
                        xaxis_title="Predicted",
                        yaxis_title="Truth",
                        xaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1],
                                        ticktext = labels
                                    ),
                        yaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1],
                                        ticktext = labels
                                    ),
                        autosize = False,
                        width = 600,
                        height = 600,
                        )
                        
        fig.update_xaxes(side="top")
        plot_conf_nb = plot(fig, output_type='div')
        
        csv_report_nb = pd.DataFrame(classification_report(orig,
                                                    preds,
                                                    output_dict=True,
                                                    
                                                    )
                                    )
        
        csv_report_nb = csv_report_nb.iloc[:-1, :].T
        csv_report_nb = csv_report_nb.round(3)
        #print(csv_report_nb)
        
        
        z_data = csv_report_nb.values.tolist()
        x_data = csv_report_nb.columns.to_list()
        y_data = csv_report_nb.index.tolist()
        

        y_data[0] = 'NonCKD'
        y_data[1] = 'CKD'
  


        fig_report = ff.create_annotated_heatmap(z_data, x_data, y_data, colorscale='blues', )
                                      
        

        fig_report.update_yaxes(autorange="reversed")
        fig_report.update_layout(title="Classification Report",
                        #xaxis_title="x Axis Title",
                        #yaxis_title="y Axis Title",
                        autosize = False,
                        width = 600,
                        height = 400,                  
                        )
        
        for i in range(len(fig_report.layout.annotations)):
                fig_report.layout.annotations[i].font.size = 15
                        
        fig_report.update_yaxes(categoryorder='category ascending')
        plot_report_nb = plot(fig_report, output_type='div')

        
        return render(request, 'administrator/NaiveBayes.html',{'Report': plot_report_nb, 
                                                                'skor_acc':Val_Score*100,
                                                                'skor_val':Train*100,
                                                                'plot_div_conf_nb': plot_conf_nb })

    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass

@login_required(login_url=settings.LOGIN_URL)
def NaiveBayesAda(request):
    if default_storage.exists('dataset_encode.csv'):
        #Target = pd.read_csv(default_storage.path('target.csv'))
        dataset = pd.read_csv('./media/dataset_encode.csv')
        X = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].values
        sc = StandardScaler()
        X = sc.fit_transform(X)
        
        X_train , X_test , y_train , y_test   = train_test_split(X,y,test_size = 0.2 , random_state=420, shuffle=True)
        
        #print(X_test)
        #print(y_test)
        #print(X_train.columns)
        
        
        Logit_Model = naive_bayes.GaussianNB()
        boost_Logit_Model = AdaBoostClassifier(base_estimator=Logit_Model,
                                               n_estimators=50,
                                               learning_rate=0.5,
                                               algorithm='SAMME.R',
                                               random_state=1
                                               )
        
        boost_Logit_Model.fit(X_train, y_train)
        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=69696969)
        n_scores_1 = cross_val_score(boost_Logit_Model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        Train = n_scores_1.mean()
        
        Prediction = boost_Logit_Model.predict(X_test)
        Val_Score = accuracy_score(y_test,Prediction)
        Report = classification_report(y_test,Prediction)

        print(Prediction)
        print("Accuracy Score: {}%".format(Val_Score*100))
        print(Report)
        
        labels=['NonCKD', 'CKD']
        preds = np.array(boost_Logit_Model.predict(X_test))
        #preds2 = Logit_Model.score(X_test, y_test)
        #preds = np.argmax(preds, axis = -1)
        orig = y_test
        conf = confusion_matrix(orig, preds)
        
        fig = ff.create_annotated_heatmap(conf, colorscale='blues', x=labels, y=labels)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
                        xaxis_title="Predicted",
                        yaxis_title="Truth",
                        xaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1],
                                        ticktext = labels
                                    ),
                        yaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1],
                                        ticktext = labels
                                    ),
                        autosize = False,
                        width = 600,
                        height = 600,
                        )
                        
        fig.update_xaxes(side="top")
        plot_conf_nb = plot(fig, output_type='div')
        
        csv_report_nb = pd.DataFrame(classification_report(orig,
                                                    preds,
                                                    output_dict=True,
                                                    
                                                    )
                                    )
        
        csv_report_nb = csv_report_nb.iloc[:-1, :].T
        csv_report_nb = csv_report_nb.round(3)
        #print(csv_report_nb)
        
        
        z_data = csv_report_nb.values.tolist()
        x_data = csv_report_nb.columns.to_list()
        y_data = csv_report_nb.index.tolist()

        y_data[0] = 'NonCKD'
        y_data[1] = 'CKD'


        fig_report = ff.create_annotated_heatmap(z_data, x_data, y_data, colorscale='blues', )
                                      
        

        fig_report.update_yaxes(autorange="reversed")
        fig_report.update_layout(title="Classification Report",
                        #xaxis_title="x Axis Title",
                        #yaxis_title="y Axis Title",
                        autosize = False,
                        width = 600,
                        height = 400,                  
                        )
        
        for i in range(len(fig_report.layout.annotations)):
                fig_report.layout.annotations[i].font.size = 15
                        
        fig_report.update_yaxes(categoryorder='category ascending')
        plot_report_nb = plot(fig_report, output_type='div')

        
        return render(request, 'administrator/NaiveBayesAda.html',{'Report': plot_report_nb, 
                                                                'skor_acc':Val_Score*100,
                                                                'skor_val':Train*100,
                                                                'plot_div_conf_nb': plot_conf_nb })

    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass



@login_required(login_url=settings.LOGIN_URL)
def hasil_nb_algen(request):
    if request.method == 'GET':
        jumlah_fitur = int(request.GET['jumlah_fitur'])
        jumlah_populasi = int(request.GET['jumlah_populasi'])
        crossover = float(request.GET['crossover'])
        mutasi = float(request.GET['mutasi'])
        jumlah_generasi = int(request.GET['jumlah_generasi'])
        jumlah_gen_no_change = int(request.GET['jumlah_gen_no_change'])
        
        rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=69420213)
        
        estimator = naive_bayes.GaussianNB()
        
        Target = pd.read_csv(default_storage.path('target.csv'))
        Features = pd.read_csv(default_storage.path('fitur.csv'))
        X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.3, random_state=52)
        
        selector = []
        
        selector = selector.fit(X_train,y_train)
        
        genfeats = X_train.columns[selector.support_]
        genfeats = list(genfeats)
        
        y_pred = selector.predict(X_test)
        
        val_akurasi = accuracy_score(y_test,y_pred)
        
        train_akurasi = selector.generation_scores_[-1]
        
        labels=['High', 'Low', 'Mid']
        preds = np.array(selector.predict(X_test))
        #preds2 = Logit_Model.score(X_test, y_test)
        #preds = np.argmax(preds, axis = -1)
        orig = y_test
        conf = confusion_matrix(orig, preds)
        
        fig = ff.create_annotated_heatmap(conf, colorscale='blues', x=labels, y=labels)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
                        xaxis_title="Predicted",
                        yaxis_title="Truth",
                        xaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1,2],
                                        ticktext = labels
                                    ),
                        yaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1,2],
                                        ticktext = labels
                                    ),
                        autosize = False,
                        width = 600,
                        height = 600,
                        )
                        
        fig.update_xaxes(side="top")
        plot_conf_nb_algen = plot(fig, output_type='div')
        
        csv_report_nb = pd.DataFrame(classification_report(orig,
                                                    preds,
                                                    output_dict=True,
                                                    
                                                    )
                                    )
        
        csv_report_nb = csv_report_nb.iloc[:-1, :].T
        csv_report_nb = csv_report_nb.round(3)
        
        z = csv_report_nb.values.tolist()
        x = csv_report_nb.columns.to_list()
        y = csv_report_nb.index.tolist()


        fig_report = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='blues', 
                                        #x=['High', 'Low', 'Mid'], y=['High', 'Low', 'Mid']
                                        )

        #fig.update_yaxes(autorange="reversed")
        fig_report.update_layout(title="Plot Title",
                        #xaxis_title="x Axis Title",
                        #yaxis_title="y Axis Title",
                        autosize = False,
                        width = 700,
                        height = 700,                  
                        )
        
        for i in range(len(fig_report.layout.annotations)):
                fig_report.layout.annotations[i].font.size = 15
                        
        fig_report.update_yaxes(categoryorder='category descending')
        plot_report_nb_algen = plot(fig_report, output_type='div')
        

        return render(request, 'administrator/hasil_nb_algen.html',{'fitur_terpilih':genfeats, 
                                                                 'val_akurasi': val_akurasi, 
                                                                 'train_akurasi': train_akurasi,
                                                                 'plot_div_conf_nb_algen': plot_conf_nb_algen,
                                                                 'plot_div_report_nb_algen': plot_report_nb_algen,
                                                                 })

    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass
