# Análise e aprendizado de máquina
 Projeto de análise a aprendizado de máquina usando data science e machine learning com Python.<br>
 Todo o código fonte está bem comentado com fins didáticos, facilitando o estudo e interpretação dos dados.


# Índice 

* [Descrição do Projeto](#descrição-do-projeto)
* [Desenvolvedor](#desenvolvedor)
* [Objetivo](#Objetivo)
* [Machine learning](#Machine-learning)
* [Métricas estatísticas utilizadas](#Métricas-estatísticas-utilizadas)
* [Modelos de machine learning utilizados](#Modelos-de-machine-learning-utilizados)
* [7 passos do machine learning](#7-passos-do-machine-learning)
* [Modelo em Produção (local host)](#Modelo-em-Produção-(local-host))
* [Inspirações e créditos](#Inspirações-e-créditos)


# Descrição do projeto

No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.

Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.

Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)

Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.

# Desenvolvedor

| [<img src="https://user-images.githubusercontent.com/115194365/202005566-f6278b6c-4f75-416f-b01c-e79b8d04f02e.jpg" width=115><br><sub>Daniel de Souza Amorim</sub>](https://github.com/DaniellsamorimGit) |
| :---: | 


#### Mais sobre o autor: <br>
Graduado em Engenharia de computação em 2010 pela Universidade Potiguar do RN;<br>
Pós-graduado em Petróleo e gás;<br>
Desenvolvedor de dispositivos embarcados, microcontrolados, automação de sistemas;<br>
Desenvolvedor de placas de CI, prototipagem e desenvolvimento;<br>
Amante de tecnologias e desenvolvimento Python.<br>

# Objetivo

Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não., Tudo isso utilizando aprendizado de máquina (machine learning) para prever ou indicar um valor sugerido.

# Machine learning

### Inicialmente temos que entender que o computador aprende de 3 formas:

- 1º aprendizado supervisionado: Quando passamos dados rotulados (um gabarito) onde temos as perguntas já com as respostas. Ex capcha de login. Esse modelo exige uma grande ***diversidade de informações. (principais métodos supervisionados: árvore de decisão, Naive bayes, regressão linear)
    
- 2º aprendizado não supervisionado: Passamos uma base sem respostas, por observação da máquina, por exemplo: Mulher compra mais maquiagem, homem compra mais ferramentas, um site vai oferecer o que mais aquele "grupo" de usúarios procura. O computador separa em grupos por características, por exemplo: ifood, algumas pessoas compram mais ou por cupom ou oferecendo frete grátis outras nem compram.
    
- 3º aprendizado por reforço: Vamos ensinando ao computador por meio de reforço positivo ou negativo, exemplo, recomendação de vídeo do youtube, o youtube mostra vídeos, se eu não assistir é pq não gosta, se assistir é pq eu gosta, assim o youtube vai aprendendo por reforço o que o usuário mais gosta.

# Métricas estatísticas utilizadas

    - Quartis 
    - R² (Erro quadratico médio)
    - RSME (raiz do erro quadrática média)
    
# Modelos de machine learning utilizados

    - Linear regression
    - Randon forest regressor
    - Extra trees

# 7 passos do machine learning

- 1º Definir se é classificação ou regressão 
        - Em resumo classificação: se é uma maçã ou abacaxi, ou regressão é o preço da maçã (específico).
        
- 2º Escolher modelo de métricas estatísticas para avaliar o modelo:
    
    ### Existem diversas métricas, usaremos 2:
    
- Primeiro R² (Erro quadratico médio): Esta métrica nos diz um percentual de acerto de 0 a 100% (maior melhor) porém acerta muito mas erra longe do valor, um R2 com 92% significa que consegue explicar 92% do acerto.
            
- RSME (raiz do erro quadrática média): Nos diz o quanto está errando (menor melhor) acerta pouco mas erra próximo do valor, por exemplo, se erra 10% significa que está errando 10% do nosso valor.
            
- 3º Escolher os modelos a usar/testar: Qual nosso problema? mas não se preocupe, inicialmente não precisamos saber a estatística por trás, temos bibliotecas Python prontas pra resolver nosso problema, inicialmente temos que "entender o modelo escolhido" pra resolver nosso problema:
    
        - Linear regression: Tenta traçar uma reta mais próxima dos acertos.
        
        - Randon forest regressor: É uma árvore de decisão, procura a melhor pergunta para dividir ao máximo o número 
        de respostas, por exemplo, adivinhar o nome de uma pessoa, Randon forest começa com a pergunta "é homem ou mulher" 
        assim já elimina 50% dos erros.
        
        - Extra trees: Parecido com radom forest porém gera perguntas aleatórias, buscando chegar mais rápido a pergunta.
        
- 4º treinar e testar os modelos: 
        - Consiste em separar a base de dados em teste e treino, por convenção dividimos em 75% treino e 25% teste (mas não é uma regra)
        - Treino são os dados passados para aprendizado
        - Teste são dados passados para conferir a acuracidade do aprendizado.
        
    OBS: cuidado com OVERFITTING: Passar todos os dados e o modelo ficar ruim para uma informação nova (vício), pois ele aprendeu o "que viu" mas não aprendeu a classificar coisas novas. Para evitar isso sempre devemos passar dados de teste e treino assim teremos uma porcentagem de acuricity, que é quem vai nos dizer o quão o modelo escolhido é acertivo.
    
    
- 5º Comparar os resultados entre varios modelos e escolher a melhor métrica (melhor acuricity):
        - Como escolher o melhor modelo?
        - calculando o R² e o RSME para cada modelo.
        e se R² e RSME tiverem o mesmo acerto? Então vamos excolher o modelo que tiver menor tempo de processamento, tipo, R² levou uma hora pra processar e RSME levou 5 minutos? então vamos escolher o menor tempo.
        DICA: modelos mais simples sempre são as melhores métricas. Exemplo: um modelo que precisa de 2 features é tem um acerto de 80% é melhor que um modelo que acerta 90% com 30 features.
    
- 6º Avaliar mais a fundo o modelo escolhido:
        - Analisar as features mais importantes que estão impactando no acerto, removendo colunas que pouco impactuam na nossa análise, ou seja, tornando nosso processamento mais rápido com pouco impacto nos resultados.
    
    
- 7º Ajustar o melhor modelo:
        - Com o modelo vencedor em mãos, fazer ajustes observando o R² e RSME que nos traga o melhor resultado.
        - Resumindo, tirou uma coluna e não impactou ou teve impacto mínimo no R² ou RSME, então exclui aquela coluna melhorando a análise qualitativa.
    
# Modelo em Produção (local host)

Por fim geramos o modelo em produçao (FRONTEND), onde os usuários podem acessar no navegador (em rede local ou disponibilizado
na internet) e preencher as opções do imóvel (features) e obter o preço com base nas opções desejadas. Os valores sugeridos
são com base no aprendizado de máquina para época do projeto, tendo em vista que, para se manter um projeto atualizado é extremamente
importante o administrador do projeto está sempre atualizando as bases de dados e para melhoria, sempre testando novas features.

https://user-images.githubusercontent.com/115194365/207130687-a92a628e-03ae-4b91-add5-50a664a38565.mp4


# Inspirações e créditos

As bases de dados foram retiradas do site kaggle: 

    - https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro

Este projeto foi feito com base no projeto americano de Allan Bruno do kaggle no Notebook: 

    - https://www.kaggle.com/allanbruno/helping-regular-people-price-listings-on-airbnb
    
Todo projeto foi refeito por DANIEL S. AMORIM, orientado pela Hashtag treinamentos, no qual eu tive a honra de participar e obter certificado Phyton.

O projeto americano serviu como inspiração, sendo modificado para a realidade no Brasil, 
com valores descritos em reais para localidade (latitude e longitude) no Rio de Janeiro e explicações em Português/BR.

- As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
- Os preços são dados em reais (R$)
- Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados

