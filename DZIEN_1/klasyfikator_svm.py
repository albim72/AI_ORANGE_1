import numpy as np
import matplotlib.pyplot as plt

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
