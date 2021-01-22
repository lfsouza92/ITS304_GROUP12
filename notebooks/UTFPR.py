import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
             break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



UC3 = pd.read_csv("C:\\Users\\Admin\\Desktop\\UC3.csv", delimiter = ';')

df_train = UC3["ENERGIA_PONTA"].iloc[:35]
df_test = UC3["ENERGIA_PONTA"].iloc[35:]

fig, ax = plt.subplots(figsize=(15, 5))
_ = plt.plot(df_test, marker='o', color="r")
_ = plt.plot(df_train, marker='o', color="b")
_ = plt.xlabel("Tempo")
_ = plt.ylabel("KW")
_ = plt.legend([ 'Dados de Test', 'Dados de Treino'])
_ = plt.title("Energia Ponta")
plt.show()


X_train, Y_train = split_sequence(df_train.values,1)
X_test, Y_test = split_sequence(df_test.values,1)

plt.scatter(X_train,Y_train)
_ = plt.title("Energia Ponta")
_ = plt.xlabel("X_train")
_ = plt.ylabel("Y_train")
plt.show()

from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, Y_train) # treinamento do modelo

print(modelo.coef_)

previsoes = modelo.predict(X_test)
df_previsoes = pd.DataFrame(previsoes,index=range(35,41))

fig, ax = plt.subplots(figsize=(15, 5))
_ = plt.plot(df_test, marker='o', color="r")
_ = plt.plot(df_train, marker='o', color="b")
_ = plt.plot(df_previsoes, marker='o', color="g")
_ = plt.xlabel("Tempo")
_ = plt.ylabel("KW")
_ = plt.legend([ 'Dados de Test', 'Dados de Treino', 'Previsões'])
_ = plt.title("Energia Ponta")
plt.show()


def RMSE(yh, y):
  er = yh - y
  RMSE = sum(er*er/len(er))**0.5
  return RMSE


print("RMSE Absoluto do Modelo = ",RMSE(previsoes, Y_test))


print("RMSE Percentual do Modelo = ",RMSE(previsoes, Y_test)/max(df_train) * 100, "%")

# #EXPERIMENTO 2

# # Vamos criar um modelo por cada mes.
# # PEGAMOS OS dados separados por meses é dizer pegamos todos os dados de janeiro dos nao 2014,2016,2017,2018,2019 prevvemos 2020 e assim para todos os meses.
# Lista_previsoes = []
# Lista_RMSE = []
# #Modelo Junho
# data = UC3[UC3['MÊS']=="Junho"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:5]
# df_test = data["ENERGIA_PONTA"].iloc[5:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_junho = LinearRegression()
# modelo_junho.fit(X_train, Y_train)
# previsoes = modelo_junho.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Junho = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Julho
# data = UC3[UC3['MÊS']=="Julho"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:3]
# df_test = data["ENERGIA_PONTA"].iloc[3:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_julho = LinearRegression()
# modelo_julho.fit(X_train, Y_train)
# previsoes = modelo_julho.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Julho = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Agosto
# data = UC3[UC3['MÊS']=="Agosto"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:4]
# df_test = data["ENERGIA_PONTA"].iloc[4:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_agosto = LinearRegression()
# modelo_agosto.fit(X_train, Y_train)
# previsoes = modelo_agosto.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Agosto = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Setembro
# data = UC3[UC3['MÊS']=="Setembro"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:4]
# df_test = data["ENERGIA_PONTA"].iloc[4:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_setembro = LinearRegression()
# modelo_setembro.fit(X_train, Y_train)
# previsoes = modelo_setembro.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Setembro = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Outubro
# data = UC3[UC3['MÊS']=="Outubro"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:4]
# df_test = data["ENERGIA_PONTA"].iloc[4:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_outubro = LinearRegression()
# modelo_outubro.fit(X_train, Y_train)
# previsoes = modelo_outubro.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Outubro = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Novembro
# data = UC3[UC3['MÊS']=="Novembro"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:4]
# df_test = data["ENERGIA_PONTA"].iloc[4:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_novembro = LinearRegression()
# modelo_novembro.fit(X_train, Y_train)
# previsoes = modelo_novembro.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Novembro = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Dezembro
# data = UC3[UC3['MÊS']=="Dezembro"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:4]
# df_test = data["ENERGIA_PONTA"].iloc[4:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_dezembro = LinearRegression()
# modelo_dezembro.fit(X_train, Y_train)
# previsoes = modelo_dezembro.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Dezembro = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Janeiro
# data = UC3[UC3['MÊS']=="Janeiro"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:5]
# df_test = data["ENERGIA_PONTA"].iloc[5:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_janeiro = LinearRegression()
# modelo_janeiro.fit(X_train, Y_train)
# previsoes = modelo_janeiro.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Janeiro = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Fevereiro
# data = UC3[UC3['MÊS']=="Fevereiro"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:5]
# df_test = data["ENERGIA_PONTA"].iloc[5:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_fevereiro = LinearRegression()
# modelo_fevereiro.fit(X_train, Y_train)
# previsoes = modelo_fevereiro.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Fevereiro = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Março
# data = UC3[UC3['MÊS']=="Março"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:5]
# df_test = data["ENERGIA_PONTA"].iloc[5:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_marzo = LinearRegression()
# modelo_marzo.fit(X_train, Y_train)
# previsoes = modelo_marzo.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Março = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Abril
# data = UC3[UC3['MÊS']=="Abril"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:5]
# df_test = data["ENERGIA_PONTA"].iloc[5:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_abril = LinearRegression()
# modelo_abril.fit(X_train, Y_train)
# previsoes = modelo_abril.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Abril = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))
# #Modelo Maio
# data = UC3[UC3['MÊS']=="Maio"][["ENERGIA_PONTA"]]
# df_train = data["ENERGIA_PONTA"].iloc[:5]
# df_test = data["ENERGIA_PONTA"].iloc[5:]
# X_train, Y_train = split_sequence(df_train.values,1)
# Y_test = df_test.values
# X_test = np.array(Y_train[-1])
# modelo_Maio = LinearRegression()
# modelo_Maio.fit(X_train, Y_train)
# previsoes = modelo_Maio.predict(X_test.reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# print("RMSE da Previsão de Maio = ", RMSE(previsoes, Y_test))
# Lista_RMSE.append(RMSE(previsoes, Y_test))


# df_train = UC3["ENERGIA_PONTA"].iloc[:50]
# df_test = UC3["ENERGIA_PONTA"].iloc[50:]
# df_previsoes = pd.DataFrame(Lista_previsoes, index=range(50,62,1))   # não sei o segundo indice

# fig, ax = plt.subplots(figsize=(15, 5))
# _ = plt.plot(df_test, marker='o', color="r")
# _ = plt.plot(df_train, marker='o', color="b")
# _ = plt.plot(df_previsoes, marker='o', color="g")
# _ = plt.xlabel("Tempo")
# _ = plt.ylabel("KW")
# _ = plt.legend([ 'Dados de Test', 'Dados de Treino', 'Previsões'])
# _ = plt.title("Energia Ponta")
# plt.show()

# print("RMSE Absoluto do Modelo = ", np.mean(Lista_RMSE))


# print("RMSE Percentual do Modelo 2= ",np.mean(Lista_RMSE)/max(df_train) * 100, "%")

# # Previsão Final do consumo para o proximo ano utilizando nosso melhor modelo
# Lista_previsoes = []
# previsoes = modelo_junho.predict(np.array(df_test.iloc[0]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_julho.predict(np.array(df_test.iloc[1]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_agosto.predict(np.array(df_test.iloc[2]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_setembro.predict(np.array(df_test.iloc[3]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_outubro.predict(np.array(df_test.iloc[4]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_novembro.predict(np.array(df_test.iloc[5]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_dezembro.predict(np.array(df_test.iloc[6]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_janeiro.predict(np.array(df_test.iloc[7]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_fevereiro.predict(np.array(df_test.iloc[8]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_marzo.predict(np.array(df_test.iloc[9]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_abril.predict(np.array(df_test.iloc[10]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])
# previsoes = modelo_Maio.predict(np.array(df_test.iloc[11]).reshape((1,1)))
# Lista_previsoes.append(previsoes[0])

# df_previsoes = pd.DataFrame(Lista_previsoes,index=range(66,78,1))

# fig, ax = plt.subplots(figsize=(15, 5))
# _ = plt.plot(df_previsoes, marker='o', color="r")
# _ = plt.plot(UC3["ENERGIA_PONTA"], marker='o', color="b")
# _ = plt.xlabel("Tempo")
# _ = plt.ylabel("KW")
# _ = plt.legend([ 'Dados Previstos', 'Dados Reais'])
# _ = plt.title("Energia Ponta")
# plt.show()