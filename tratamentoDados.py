from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from  keras.utils import to_categorical
import numpy as np

class TratamentoDados():
    
    def codidifca_label_encoder(self, x):
        return LabelEncoder().fit_transform(x)
    
    def codifica_one_hot_encoder(self, x):
        return to_categorical(x)
    
    #Normalização:
    def padroniza(self, x):
        x = (x - np.mean(x))/np.std(x)
        return x
    
    # Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)
    def split(self, x, y, tamTreino):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = tamTreino, shuffle =False, random_state=3)
        return x_train, x_test, y_train, y_test
    
    def covert_to_binary(self, y_pred):
        for i in range(y_pred.shape[0]):
            y_pred[i,0] = 1 if y_pred[i,0] > 0.5 else 0
        return y_pred