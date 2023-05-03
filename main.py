import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("advertising.csv")
valoriX = np.asarray(df.TV.values)
valoriY = np.asarray(df.Sales.values)

x = [[0 for _ in range(2)] for _ in range(len(valoriX))]  #Matricea x goala
y = np.array(valoriY)  #Crearea vectorului a valorilor lui y pentru a putea calcula parametrii a si b

for i in range(0, len(valoriX)):  #Crearea matricii a valorilor lui x pentru a putea calcula parametrii a si b
    x[i][0] = 1
    x[i][1] = valoriX[i]

  #Calculul parametrii a si b utilizand metoda celor mai mici patrate
xtranspusa = np.transpose(x)
first = np.matmul(xtranspusa, x)
inversa = np.linalg.inv(first)
final = np.matmul(inversa, xtranspusa)
final = np.matmul(final, y)

  #Calculul pentru reziduuri
reziduuri = final[0]+final[1]*np.array(valoriX) - np.array(valoriY)


#Calculul pentru media patratelor reziduurilor
MSE = np.mean(np.square(reziduuri))
print(MSE)

#Afisarea diagramei pentru regresia liniara
plt.scatter(valoriX, valoriY, marker='x', color='green') #Afisare valorilor x si y pe diagrama
plt.plot(valoriX, final[0] + final[1] * np.array(valoriX), color="r") #Formula pentru a afisa dreapta de regresie
plt.text(90, -3, 'Numarul vanzarilor', fontsize=15, color = "blue")
plt.text(-55, 5, 'Numarul de pauze publicitare TV', rotation=90, fontsize=15, color = "blue")

#Afisarea histogramei pentru reziduuri
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
sns.histplot(data=reziduuri, color="blue", binrange=(-55, 55), bins=100, ax=ax2, kde = True)
ax2.set_xlabel('Reziduuri', color='blue')
plt.show()

