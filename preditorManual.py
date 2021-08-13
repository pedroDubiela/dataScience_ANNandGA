import pandas as pd
import numpy as np
import tratamentoDados
from tratamentoDados import TratamentoDados as trd
from sklearn.metrics import accuracy_score

def trataDados():
    
     #importação dos dados:
     dados = pd.read_csv("diabetes_data_upload.csv")
     x = dados.iloc[:,0:16].values
     y = dados.iloc[:,16].values
     
     #Codificação label para todos os atributos categoricos, exceto a idade:
     for i in range(1,16):
         x[:,i]  = td.codidifca_label_encoder(x[:,i])
    
     
     #Codificação label para o atributo meta:
     y =  td.codidifca_label_encoder(y)
    
     #Padronização da idade:
     x[:,0] = td.padroniza(x[:,0])
 
     #Split:
     return td.split(x, y, 0.3)
    

def funcoesModelo(x_train, numNeuroC1, numeroEntradas):
    
    #Sinapses e Saída: primeira camada oculta:
    lin_max = x_train.shape[0]
    sinapses1 = np.empty(shape = (lin_max, numNeuroC1))
    saidas1 = np.empty(shape = (lin_max, numNeuroC1))
    
    #Sinapses e Saída: ultima camada:
    sinapses2 = np.empty(shape = ( lin_max,1))
    saidas2 = np.empty(shape = ( lin_max,1))
    
    #Funcao de ativação:
    reLU = lambda x: x if x >0 else 0
    sigmoidal = lambda x: 1/(1+np.exp(-x))
    
    return lin_max, sinapses1, saidas1, sinapses2, saidas2, reLU, sigmoidal
    
def feedForward(epochs, x, y, W1, W2, lin_max, sinapses1, saidas1, sinapses2, saidas2, reLU, sigmoidal):
    
    #FeedForward
    for n in range(epochs):
        for i in range(lin_max):
            
            #Para a camada 1:
            sinapses1[i, :] = np.dot(W1, x[i,:])
            #Saindo da camada 1:
            saidas1[i, :] = list(map(reLU, sinapses1[i,:]))
            #Para a ultima camada:
            sinapses2[i, :] = np.dot(saidas1[i, :], W2)
            #Saindo da ultima camada:
            saidas2[i, :] = list(map(sigmoidal , sinapses2[i,:]))
            
        #Resultado
        y_pred = td.covert_to_binary(saidas2)
        taxaAcerto = accuracy_score(y_pred, y)
        
    return y_pred, taxaAcerto
      
class Individuo():
    
    def __init__(self, numeroEntradas, numNeuroC1, geracao = 0):
        self.numeroEntradas = numeroEntradas
        self.numNeuroC1 = numNeuroC1
        self.geracao = geracao
        self.nota = 0
        self.saida = 0
        self.cromossomo =[np.random.uniform(-3.74, 1.57,(numeroEntradas, numNeuroC1)),
                          np.random.uniform(-3.25, 1.17,(numNeuroC1, 1))]

    def avaliacao(self, epochs, x_train, y_train, lin_max,
                  sinapses1, saidas1, sinapses2,
                  saidas2, reLU, sigmoidal):
        
        (self.saida, self.nota) = feedForward(epochs, x_train, y_train, 
                    self.cromossomo[0], self.cromossomo[1], lin_max, sinapses1,
                    saidas1, sinapses2, saidas2, 
                    reLU, sigmoidal)
        
    def crossOver(self, outro, taxaMutacao):
        
        j = int(np.random.randint(0, self.cromossomo[0].shape[1], size = 1))
        i = int(np.random.randint(0, self.cromossomo[1].shape[0], size = 1))
        
        #Filho1:
        #W1 corte na coluna junção horizontal
        filho1W1 = np.hstack((self.cromossomo[0][:,0:j], outro.cromossomo[0][:,j::]))
        
        #W2 corte na linha junção vertical
        filho1W2 = np.vstack((self.cromossomo[1][0:i], outro.cromossomo[1][i::]))
        
        #Filho2:
        #W1 corte na coluna junção horizontal
        filho2W1 = np.hstack((outro.cromossomo[0][:,0:j], self.cromossomo[0][:,j::]))
        
        #W2 corte na linha junção vertical
        filho2W2 = np.vstack((outro.cromossomo[1][0:i], self.cromossomo[1][i::]))
        
        filhos = [Individuo(self.numeroEntradas, self.numNeuroC1, self.geracao + 1),
                  Individuo(self.numeroEntradas, self.numNeuroC1, self.geracao + 1)]
        
        #Mutação:
        aleatorio = np.random.random()
        if aleatorio  < taxaMutacao:
            filho1W1 = filho1W1 * (-1)
            filho1W2 = filho1W2 * (-1)
            filho2W1 = filho2W1 * (-1)
            filho2W2 = filho2W2 * (-1)
        
        filhos[0].cromossomo = [filho1W1, filho1W2]
        filhos[1].cromossomo = [filho2W1, filho2W2]
        
        
        return filhos
    
 
        
class Populacao():
    
    def __init__(self, tamanho_populacao):
        self.tamanho_populacao = tamanho_populacao
        self.populacao = []
        self.geracao = 0
        self.melhor_solucao = 0
        self.lista_solucao = []
        
    def inicializaPopulacao(self, numeroEntradas, numNeuroC1):
        for i in range(self.tamanho_populacao):
            self.populacao.append(Individuo(numeroEntradas, numNeuroC1))
        self.melhor_solucao = self.populacao[0]

            
    def avaliaPopulacao(self, epochs, x_train, y_train, lin_max,
                        sinapses1, saidas1, sinapses2, saidas2, reLU, sigmoidal):
        for individuo in self.populacao:
            individuo.avaliacao(epochs, x_train, y_train, lin_max, sinapses1,
                                saidas1, sinapses2, saidas2, reLU, sigmoidal)
            
    def ordenaPopulacao(self):
        self.populacao = sorted(self.populacao, 
                               key = lambda individuo: individuo.nota,
                               reverse = True)
        self.lista_solucao.append(self.populacao[0])
        
    def melhorIndividuoGeral(self):
        if self.melhor_solucao.nota < self.populacao[0].nota:
            self.melhor_solucao = self.populacao[0]
            
            
    def pondera(self):
        soma = 0
        
        for individuo in self.populacao:
            soma += individuo.nota
            
        pesos = []
        
        for individuo in self.populacao:
            pesos.append(individuo.nota/soma)
            
        return pesos
    
    def selecionaPai(self):
        posicao_pais = list(range(0,len(self.populacao)))
        pesos = self.pondera()
        pai_id = np.random.choice(a = posicao_pais,
                                  size = 1,
                                  p = pesos)
        return pai_id



if __name__ == '__main__':
    
    np.random.seed(1)
    
    #Módulo para tratamendo dos dados:
    td = trd()
    
    #Parametros Fixos:
    epochs = 100
    geracao = epochs
    taxaMutacao = 0.01
    
    #Dados prontos para serem utilizados pela rede neural:
    x_train, x_test, y_train, y_test = trataDados()
    
    #Entradas:
    numNeuroC1 = 16
    numeroEntradas = x_train.shape[1]
    
    #Modelo:
    (lin_max, sinapses1, saidas1, sinapses2, saidas2, reLU, sigmoidal) = funcoesModelo(x_train,
                                                                                       numNeuroC1,
                                                                                       numeroEntradas)
    #Algoritmo:
    #Instanciado a classe População:
    tamanho_populacao = 20
    pop = Populacao(tamanho_populacao)
    
    #Inicializando a População:
    pop.inicializaPopulacao(numeroEntradas ,numNeuroC1)
    
    for n in range(geracao):
        print(f'geracao = epoch = {n}')
        #Avaliação:Aqui está a rede neural
        pop.avaliaPopulacao(1, x_train, y_train, lin_max, sinapses1, saidas1, sinapses2, saidas2, reLU, sigmoidal)
        
        #Ordena população:
        pop.ordenaPopulacao()
        
        #Melhor resposta geral:
        pop.melhorIndividuoGeral()
       
        #Gerando Nova População:
        nova_populacao = []
        for numero_cross in range(0, len(pop.populacao),2):
            
            #Seleção Pais
            pai_id1 = int(pop.selecionaPai())
            pai_id2 = int(pop.selecionaPai())
            
            #Crossover
            filhos = pop.populacao[pai_id1].crossOver(pop.populacao[pai_id2], taxaMutacao)
            
            #Mutação:COlocar
            nova_populacao.append(filhos[0])
            nova_populacao.append(filhos[1])
        
        pop.populacao = nova_populacao
        pop.geracao += 1
    
    
    #Modelo:
    (lin_max, sinapses1, saidas1, sinapses2, saidas2, reLU, sigmoidal) = funcoesModelo(x_test,
                                                                                       numNeuroC1,
                                                                                       numeroEntradas)
    
    (y_pred, taxaAcerto) = feedForward(1, x_test, y_test, pop.melhor_solucao.cromossomo[0],
                pop.melhor_solucao.cromossomo[1], x_test.shape[0], sinapses1, 
                saidas1, sinapses2, saidas2, reLU, sigmoidal)
    
    print("\n")
    print(f'Geração do melhor indivíduo: {pop.melhor_solucao.geracao}')
    print(f'Acurácia no treino do melhor indivíduo: {pop.melhor_solucao.nota}')
    print(f'Acurácia no teste do melhor indivíduo: {taxaAcerto}')
    
    
    
        
        
        
    

