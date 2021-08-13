# Objetivos:
- Criar uma rede neural do tipo Perceptron Multicamadas utilizando algoritmo genético para otimizar os pesos da rede neural, ao invés de utilizar métodos baseados na descida do gradiente.
- Testar o modelo na mesma base utilizada no repositório: https://github.com/pedroDubiela/dataScience_previsaoDiabetes
# A Rede Neural:
A rede é composta por uma camada de entrada com 16 atributos previsores, uma camada escondida e uma camada de saída. A camada escondida possui 16 neurônios e cada um utiliza a função de ativação reLU.  A camada de saída possui um único neurônio e utiliza a função de ativação sigmoide.
A rede foi criada de forma manual, sem o uso de bibliotecas, tais como, Keras, TensorFlow ou PyTorch.

# O Algoritmo Genético:
Escolheu-se trabalhar com uma população de 20 indivíduos a uma taxa de mutação de 1%. 
Devido ao alto custo computacional do modelo, utilizou-se somente 100 gerações (epochs).  

# Resultado:
A acurácia no treinamento foi de 86%, já no teste, a acurácia foi de 82%. O que mostra um pequeno efeito de overfit do modelo. A acurácia deste modelo no teste, é inferir a do modelo baseado na descida estocástica do gradiente (99% - disponível em https://github.com/pedroDubiela/dataScience_previsaoDiabetes ) 
Portanto, o modelo de rede neural baseado no algoritmo genético para otimização dos pesos, mostrou-se ser inferior à mesma arquitetura de rede neural baseada na descida estocástica do gradiente para otimização de pesos.

