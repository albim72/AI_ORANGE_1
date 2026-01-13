import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,metrics,svm
from sklearn.model_selection import train_test_split

# --- 1) Wykres funkcji sigmoid ---
x = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-x))

plt.figure(figsize=(8, 4.5))
plt.plot(x, sigmoid, label=r"$\sigma(x)=\frac{1}{1+e^{-x}}$")
plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Funkcja sigmoidalna (sigmoid)")
plt.xlabel("x")
plt.ylabel(r"$\sigma(x)$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --- 2) Zapis wzoru do PNG (pełna notacja matematyczna) ---
formula = r"$\sigma(x)=\frac{1}{1+e^{-x}}$"

fig = plt.figure(figsize=(6, 2), dpi=200)
fig.patch.set_facecolor("white")
plt.axis("off")
plt.text(
    0.5, 0.5,
    formula,
    ha="center", va="center",
    fontsize=28
)
# Zapis do pliku PNG
output_path = "sigmoid_wzor.png"
plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)

print("Wzór matematyczny funkcji sigmoid:")
print("σ(x) = 1 / (1 + e^{-x})")
print(f"Zapisano obraz ze wzorem do pliku: {output_path}")


#ładowanie danych ze zbioru digits
digits = datasets.load_digits()

#wizualizacja 4 pierwszych obrazów...
_, axes = plt.subplots(nrows=1,ncols=4,figsize=(10,3))
for ax,image,label in zip(axes,digits.images,digits.target):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title(f'Cyfra: {label}')
plt.show()

#liczba próbek
n_samples = len(digits.images)
print(n_samples)

#przeksztsałcenie macierzy obrazu w wektor 8x8 = 64
data = digits.images.reshape((n_samples,-1))

#budowa klasyfikatora
clf = svm.SVC(gamma=0.001, C=100.)
X_trian,X_test,y_trian,y_test = train_test_split(data,digits.target,test_size=0.5,random_state=0,
                                                 shuffle = False)
#trening modelu
clf.fit(X_trian,y_trian)

predicted = clf.predict(X_test)

    #wizualizacja 4 pierwszych obrazów...
_, axes = plt.subplots(nrows=1,ncols=4,figsize=(10,3))
for ax,image,prediction in zip(axes,X_test,predicted):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title(f'Znaleziono: {prediction}')
plt.show()

#ocena modelu

print(f"Rport klasyfikacyjny dla kalsyfikatora clf: {clf}"
      f"\n{metrics.classification_report(y_test,predicted)}\n")

Printer = metrics.ConfusionMatrixDisplay.from_predictions(y_test,predicted)

#macierz pomyłek
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test,predicted)
disp.plot()
plt.show()
