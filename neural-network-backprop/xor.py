import numpy as np

from neural_network import NeuralNetwork

# =============================================================================
# --- Problema 1: XOR ---
# =============================================================================

print("="*50)
print("--- Problema 1: XOR ---")
print("="*50)

# Entradas (X) e Saídas Esperadas (y)
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

# Estrutura: 2 entradas, 3 neurônios ocultos, 1 saída
nn_xor = NeuralNetwork(layer_sizes=[2, 3, 1])

# Treinamento
# *** CORREÇÃO: Aumentada a taxa de aprendizado para 1.0 para convergir ***
nn_xor.train(X_xor, y_xor, epochs=10000, learning_rate=1.0)

# Teste (Prints para o relatório)
print("\n--- Resultados (Prints) do XOR ---")
predictions_xor = nn_xor.predict(X_xor)

print(f"+-----------+----------+---------------+--------------+")
print(f"| Entrada   | Esperado | Previsto (Raw) | Arredondado  |")
print(f"+-----------+----------+---------------+--------------+")
for i in range(len(X_xor)):
    entrada_str = str(X_xor[i])
    esperado_str = str(y_xor[i][0])
    previsto_raw = predictions_xor[i][0]
    previsto_round = round(previsto_raw)
    
    print(f"| {entrada_str:<9} | {esperado_str:<8} | {previsto_raw:<13.8f} | {previsto_round:<12.0f} |")
print(f"+-----------+----------+---------------+--------------+")


# =============================================================================
# --- Problema 2: Display 7 Segmentos ---
# =============================================================================

print("\n\n" + "="*50)
print("--- Problema 2: Display 7 Segmentos ---")
print("="*50)

# Dados de entrada (Segmentos Ativos)
# Ordem dos segmentos (baseado no diagrama): a, b, c, d, e, f, g
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

# Estrutura: 7 entradas, 5 ocultos, 10 saídas (baseado na tabela one-hot)
# *** CORREÇÃO: Removida a linha com o que causou o NameError ***
nn_digits = NeuralNetwork(layer_sizes=[7, 5, 10])

# Treinamento
# *** CORREÇÃO: Trocado o primeiro 'y_digits' por 'X_digits' ***
nn_digits.train(X_digits, y_digits, epochs=15000, learning_rate=0.05)

# Teste (Prints para o relatório)
print("\n--- Resultados (Prints) dos 7 Segmentos (Sem Ruído) ---")
predictions_digits = nn_digits.predict(X_digits)

print(f"+-----------------+----------+----------+--------+")
print(f"| Entrada         | Esperado | Previsto | Status |")
print(f"+-----------------+----------+----------+--------+")
total_acertos = 0
for i in range(len(X_digits)):
    # A previsão é o índice (dígito) com o maior valor de ativação
    predicted_digit = np.argmax(predictions_digits[i])
    expected_digit = np.argmax(y_digits[i])
    status = "Acerto" if predicted_digit == expected_digit else "ERRO"
    if predicted_digit == expected_digit:
        total_acertos += 1
    
    print(f"| Dígito {expected_digit} (input) | {expected_digit:<8} | {predicted_digit:<8} | {status:<6} |")
print(f"+-----------------+----------+----------+--------+")
print(f"Acurácia (Treino): {total_acertos / len(X_digits):.1%}")


# =============================================================================
# --- Problema 2.1: Teste com Ruído ---
# =============================================================================

def add_noise(X, noise_level=0.1):
    """
    Adiciona ruído aos dados de entrada (simulando falha de segmento).
    noise_level: A probabilidade de cada segmento (bit) ser "virado".
    """
    X_noisy = X.copy()
    np.random.seed(99) # Seed diferente para o ruído
    
    for i in range(X_noisy.shape[0]):
        for j in range(X_noisy.shape[1]):
            if np.random.rand() < noise_level:
                # Vira o bit (0 -> 1, 1 -> 0)
                X_noisy[i, j] = 1 - X_noisy[i, j]
    return X_noisy

print("\n\n--- Teste de Robustez com Ruído (15%) ---")

# Criar dados de teste com ruído
noise_rate = 0.15
X_noisy = add_noise(X_digits, noise_level=noise_rate)

# Fazer previsões com o modelo já treinado
predictions_noisy = nn_digits.predict(X_noisy)

print(f"+----------+-----------------+-----------------+----------+----------+--------+")
print(f"| Esperado | Entrada Original  | Entrada Ruidosa   | Previsto | Status   |")
print(f"+----------+-----------------+-----------------+----------+----------+--------+")
total_acertos_ruido = 0
for i in range(len(X_noisy)):
    predicted_digit = np.argmax(predictions_noisy[i])
    expected_digit = np.argmax(y_digits[i])
    status = "Acerto" if predicted_digit == expected_digit else "ERRO"
    if predicted_digit == expected_digit:
        total_acertos_ruido += 1
        
    print(f"| {expected_digit:<8} | {str(X_digits[i]):<17} | {str(X_noisy[i]):<17} | {predicted_digit:<8} | {status:<6} |")
print(f"+----------+-----------------+-----------------+----------+----------+--------+")
print(f"Acurácia (Ruído {noise_rate*100}%): {total_acertos_ruido / len(X_noisy):.1%}")