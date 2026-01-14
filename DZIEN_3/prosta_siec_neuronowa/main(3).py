import numpy as np
from simplenn import SimpleNeuralNetwork


network = SimpleNeuralNetwork()
print(network)

#zestaw danych treningowych
train_inputs = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,0],[0,0,0],[0,0,1],[0,1,1]])
#zestaw etykiet
train_outputs = np.array([[1,0,1,1,1,1,0,0]]).T
#liczba epok
train_iters = 50_000

#trening sieci neuronowej
network.train(train_inputs,train_outputs,train_iters)
print(f"wytrenowane wagi:\n {network.weights}")

#budowa zbioru testowego
test_data = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])

print("________ predykcja __________")
for data in test_data:
    print(f"wynik dla {data} wynosi: {network.propagation(data)}")
