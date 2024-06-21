import json
import io
import csv
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .models import ClassificationModel, ClusteringModel

classification_model_list = ['lr', 'svm', 'nb', 'xgb', 'lstm']
clustering_model_list = ['sm', 'cos']

classification_model_obj = ClassificationModel(classification_model_list[0])
clustering_model_obj = ClusteringModel(clustering_model_list[0])


def index(request):
    return render(request, 'index.html')


def submit_form(request):
    if request.method == 'POST':
        product_title = request.POST.get('product_title')
        try:
            class_pred = classification_model_obj.predict(product_title)
            clus_pred = clustering_model_obj.predict(product_title)
            return JsonResponse(
                {'class_pred': class_pred[0], 'clus_pred': clus_pred, 'product_title': product_title})

        except Exception as e:
            print(e)

        return JsonResponse({'message': 'Some Error'})


def model_selection(request):
    if request.method == 'POST':
        model_name_dict = json.loads(request.body)
        classification_model_obj.change_model(classification_model_list[model_name_dict['classification_model']])
        clustering_model_obj.change_model(clustering_model_list[model_name_dict['clustering_model']])
        return HttpResponse('')
