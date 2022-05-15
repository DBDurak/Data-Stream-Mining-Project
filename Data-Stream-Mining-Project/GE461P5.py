
from sklearn import datasets
from skmultiflow.data import HyperplaneGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP
from skmultiflow.lazy import KNNClassifier as KNN
from skmultiflow.trees import HoeffdingTreeClassifier as HT
from sklearn.ensemble import VotingClassifier as VC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skmultiflow.meta import BatchIncrementalClassifier
from skmultiflow.evaluation import EvaluatePrequential as EP
from skmultiflow.data.file_stream import FileStream as FS
#writing the necessary functions
mlp=MLP(hidden_layer_sizes=(100,4),max_iter=10000)
knn = KNN()
ht = HT()
MV = VC(estimators=[("HT", ht), ("KNN", knn), ("MLP", mlp)], voting='hard', weights=[1,1,1])
MV= BatchIncrementalClassifier(base_estimator=MV, n_estimators=3)
WMV = VC(estimators=[("HT", ht), ("KNN", knn), ("MLP", mlp)], voting='hard')
WMV= BatchIncrementalClassifier(base_estimator=WMV, n_estimators=3)
noise =[0.1,0.3]
d_feat = [2,5]

#dataset generation
for i in noise:
    for j in d_feat:
        hg = HyperplaneGenerator(noise_percentage=i, n_drift_features=j, random_state=1)
        hg = hg.next_sample(20000)
        hg_save = np.append(hg[0],np.reshape(hg[1],(20000,1)),axis=1)
        hgframe = pd.DataFrame(hg_save)
        hgframe = hgframe.astype({10: int})
        hgframe.to_csv('hg_'+str(i)+'_'+str(j)+'.csv',index=False)
        break

#reading the dataset and making the required calculations for online learning
list=[]
for i in noise:
    for j in d_feat:
        x=FS('hg_'+str(i)+'_'+str(j)+'.csv')
        evaluate=EP(show_plot=True,metrics=['accuracy'])
        
        evaluate.evaluate(stream =x, model =[mlp,knn,ht,WMV,MV], model_names=['MLP','KNN','HT','WMV','MV'])
        batch_size=[10,20,50,100,200,500,1000]
        for k in batch_size:
            print("Evaluation for Batch Size:",k ,"for Noise:",i,"and Dimensionality:",j)
            evaluate=EP(show_plot=True,metrics=['accuracy'],batch_size=k)
            a=evaluate.evaluate(stream =x, model =[mlp,knn,ht,WMV,MV], model_names=['MLP','KNN','HT','WMV','MV'])
            list.append(a)
list_to_csv=pd.DataFrame(list)
list_to_csv.to_csv('Data for Online part.csv',index=False)

            
#reading the data to an array and doing batch learning 

list2=[]
functions = (mlp,knn,ht,MV,WMV)
for i in noise:
    for j in d_feat:
        datasethg=pd.read_csv('hg_'+str(i)+'_'+str(j)+'.csv').values
        features = datasethg[:,:10]
        labels= np.array(datasethg[:,10],dtype=int)
        #splitting the data into train and test
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=1)
        for k in functions:
            k.fit(train_features,train_labels)
            pred = k.predict(test_features)
            y=acc = accuracy_score(test_labels,pred)
            print("Accuracy for batch",k, "for dataset with Noise:",i,"and Dimensionality:",j,"is:",acc)
            list2.append(y)
list2_to_csv=pd.DataFrame(list2)
list2_to_csv.to_csv('Data for Batch part.csv',index=False)
#reading the data from the csv file and plotting the graphs
from skmultiflow.meta import AdaptiveRandomForestClassifier as ARF
from skmultiflow.meta import DynamicWeightedMajorityClassifier as DWM
list3=[]
for i in noise:
    for j in d_feat:
        rfc = ARF()
        mlp = MLP(hidden_layer_sizes=(100,4),max_iter=10000)
        dwc = DWM(mlp)
        x = FS('hg_'+str(i)+'_'+str(j)+'.csv')
        evaluate=EP(show_plot=True,metrics=['accuracy'])
        z=evaluate.evaluate(stream =x, model =[rfc,mlp,dwc], model_names=['Random Forest Classifier','MLP','Dynamic Weighted Majority Classifier'])
        list3.append(z)
list3_to_csv=pd.DataFrame(list3)
list3_to_csv.to_csv('Data for Improvement part.csv',index=False)
        
        
        
        
        
