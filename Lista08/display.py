import numpy as np

# (Reutilizando a classe NeuralNetwork e funções auxiliares definidas acima)

# --- Execução do Problema 2: 7 Segmentos  ---

print("\n\n--- Problema 2: Display 7 Segmentos ---")

# Dados de entrada (Segmentos Ativos) 
# Ordem dos segmentos (baseado no diagrama [cite: 18, 19, 20, 21]): a, b, c, d, e, f, g
X_digits = np.array([
    [1, 1, 1, 1, 1, 1, 0], # 0
    [0, 1, 1, 0, 0, 0, 0], # 1
    [1, 1, 0, 1, 1, 0, 1], # 2
    [1, 1, 1, 1, 0, 0, 1], # 3
    [0, 1, 1, 0, 0, 1, 1], # 4
    [1, 0, 1, 1, 0, 1, 1], # 5
    [1, 0, 1, 1, 1, 1, 1], # 6
    [1, 1, 1, 0, 0, 0, 0], # 7
    [1, 1, 1, 1, 1, 1, 1], # 8
    [1, 1, 1, 1, 0, 1, 1]  # 9
])

# Saídas esperadas (One-hot Output) 
y_digits = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 9
])

# Estrutura: 7 entradas , 5 ocultos , 10 saídas (baseado na tabela )
nn_digits = NeuralNetwork(layer_sizes=[7, 5, 10])

# Treinamento
nn_digits.train(y_digits, y_digits, epochs=15000, learning_rate=0.05)

# Teste (Prints para o relatório)
print("\n--- Resultados (Prints) dos 7 Segmentos  ---")
predictions_digits = nn_digits.predict(X_digits)

for i in range(len(X_digits)):
    # A previsão é o índice (dígito) com o maior valor de ativação
    predicted_digit = np.argmax(predictions_digits[i])
    expected_digit = np.argmax(y_digits[i])
    status = "Acerto" if predicted_digit == expected_digit else "ERRO"
    
    print(f"Entrada: Digito {expected_digit}")
    print(f"  Previsto: {predicted_digit} ({status})")
    print(f"  Saída Raw (One-Hot): {predictions_digits[i]}") # (Descomente para ver a saída completa)