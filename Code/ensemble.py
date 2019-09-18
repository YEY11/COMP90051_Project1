# votting
# v1: nb + svc + knn1 = 0.30100
# v2: nb + svc + mlr = 0.30100
# v3: nb + svc + knn1 +mlr = 0.21898

import pandas as pd 
from collections import Counter

nb_result = pd.read_csv('../Result/result_nb.csv', encoding='UTF-8') 
svc_result = pd.read_csv('../Result/result_svc.csv', encoding='UTF-8')
knn1_result = pd.read_csv('../Result/result_knn1.csv', encoding='UTF-8')
mlr_result = pd.read_csv('../Result/result_knn1.csv', encoding='UTF-8')


nb_pred = nb_result['Predicted'].tolist()
svc_pred = svc_result['Predicted'].tolist()
knn1_pred = knn1_result['Predicted'].tolist()
mlr_pred = mlr_result['Predicted'].tolist()

nb_score = 0.20534
svc_score = 0.30128
knn1_score = 0.15012
mlr_score = 0.17100

ensemble_preds = []

for i in range(len(nb_pred)):
    nb = Counter({nb_pred[i]:nb_score})
    svc = Counter({svc_pred[i]:svc_score})
    knn1 = Counter({knn1_pred[i]:knn1_score})
    mlr = Counter({mlr_pred[i]:mlr_score})
    ensemble = nb + svc + mlr
    #print(ensemble)
    ensemble_preds.append(ensemble.most_common(1)[0][0])

id_list = list(range(1,35438))
result = pd.DataFrame(list(zip(id_list,ensemble_preds)),columns =['Id','Predicted'])
result.to_csv('../Result/result_ensemble_v3.csv',index=False)






    


