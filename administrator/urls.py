from django.urls import path, include
from . import views

app_name = 'administrator'

urlpatterns = [
    path('', views.index, name='index'),
    path('dataset/', views.dataset, name='dataset'),
    path('tentang/', views.tentang, name='tentang'),
    path('EDA/', views.EDA, name='EDA'),
    path('preproses/', views.preproses, name='preproses'),
    #path('klasifikasi/', views.klasifikasi, name='klasifikasi'),
    path('NaiveBayes/', views.NaiveBayes, name='NaiveBayes'),
    path('NaiveBayesAda/', views.NaiveBayesAda, name='NaiveBayesAda'),
    path('InfoGainR/', views.InfoGainR, name='InfoGainR'),
    path('Diskritisasi/', views.Diskritisasi, name='Diskritisasi'),
    path('NB_Custom/', views.NB_Custom, name='NB_Custom'),
    #path('hasilsvmrbf/', views.hasilsvmrbf, name='hasilsvmrbf'),
    #path('SVMRBFIG/', views.SVMRBFIG, name='SVMRBFIG'),
    path('Rangkuman/', views.Rangkuman, name='Rangkuman'),

]
