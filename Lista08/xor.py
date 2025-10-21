import numpy as np

# =============================================================================
# --- Definições Globais (Funções e Classe da Rede Neural) ---
# =============================================================================

def sigmoid(x):
    """Função sigmoide"""
    # Adicionado clip para evitar overflow/underflow em exp(-x)
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """Derivada da função sigmoide"""
    s = sigmoid(x)
    return s * (1 - s)

def mean_squared_error(y_true, y_pred):
    """Função de erro quadrático médio (MSE)"""
    return np.mean((y_true - y_pred)**2)

class NeuralNetwork:
    """
    Implementação da Rede Neural com Backpropagation a partir do zero.
    """
    def __init__(self, layer_sizes, seed=42):
        """
        Inicializa os pesos e biases.
        layer_sizes: Lista com o número de neurônios em cada camada.
                     Ex: [2, 3, 1] para 2 entradas, 3 ocultos, 1 saída.
        seed: Semente para reprodutibilidade.
        """
        np.random.seed(seed)
        self.layers = layer_sizes
        self.weights = []
        self.biases = []

        # Inicializa pesos e biases entre as camadas
        for i in range(len(layer_sizes) - 1):
            # Pesos inicializados com valores aleatórios pequenos (distribuição normal)
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            # Biases inicializados com zeros
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def feedforward(self, X):
        """Propaga a entrada (X) pela rede (fase de inferência)"""
        activations = [X] # Armazena as ativações (saídas) de cada camada
        inputs = []       # Armazena as entradas ponderadas (net) de cada camada
        
        a = X
        for i in range(len(self.weights)):
            # Cálculo da entrada ponderada (net): z = a * W + b
            z = np.dot(a, self.weights[i]) + self.biases[i]
            inputs.append(z)
            
            # Cálculo da ativação (saída da camada): a = sigmoid(z)
            a = sigmoid(z)
            activations.append(a)
            
        # Retorna:
        # a: predição final (ativação da última camada)
        # activations: lista de todas as ativações (incluindo entrada)
        # inputs: lista de todas as entradas ponderadas (net)
        return a, activations, inputs

    def backpropagate(self, X, y, learning_rate):
        """Realiza uma passagem de feedforward e backpropagation para atualizar os pesos"""
        
        # 1. Feedforward: Obtém as saídas e valores intermediários
        y_pred, activations, inputs = self.feedforward(X)
        
        # 2. Calcular o erro (MSE) para esta passagem (opcional, bom para logging)
        loss = mean_squared_error(y, y_pred)
        
        # --- Início do Backpropagation ---
        
        # 3. Calcular o delta (erro) da camada de saída
        # (y_true - y_pred) é a derivada do MSE (E) em relação à ativação (y_pred)
        # sigmoid_derivative(inputs[-1]) é a derivada da ativação (y_pred) em relação à entrada ponderada (net)
        # delta_k = (y_k - \hat{y}_k) * \sigma'(\text{net}_k)
        delta_output = (y - y_pred) * sigmoid_derivative(inputs[-1])
        
        deltas = [delta_output]
        
        # 4. Propagar o delta para as camadas ocultas (de trás para frente)
        delta_hidden = delta_output
        # Itera das penúltima camada (índice len(self.layers) - 2) até a primeira camada oculta (índice 1)
        for i in range(len(self.layers) - 2, 0, -1):
            # delta_j = (\sum_k \delta_k w_{jk}) * \sigma'(\text{net}_j)
            # Em forma vetorial: (delta_next @ W_next.T) * sigmoid_derivative(net_hidden)
            delta_hidden = np.dot(delta_hidden, self.weights[i].T) * \
                           sigmoid_derivative(inputs[i-1])
            deltas.insert(0, delta_hidden) # Adiciona o delta no início da lista
            
        # 5. Atualizar pesos e biases
        for i in range(len(self.weights)):
            # O gradiente do peso é a ativação da camada anterior multiplicada pelo delta da camada atual
            # \Delta w_{ij} = \eta \cdot \delta_j \cdot x_i
            # Em forma vetorial: activation_prev.T @ delta_current
            grad_weights = np.dot(activations[i].T, deltas[i])
            
            # O gradiente do bias é simplesmente o delta (ou a soma dele, se for em batch)
            grad_biases = np.sum(deltas[i], axis=0, keepdims=True)
            
            # Atualização dos pesos e biases na direção oposta ao gradiente
            self.weights[i] += grad_weights * learning_rate
            self.biases[i] += grad_biases * learning_rate
            
        return loss

    def train(self, X, y, epochs, learning_rate):
        """Treina a rede por um número de épocas"""
        print(f"Iniciando treinamento por {epochs} épocas (Taxa de Aprendizado: {learning_rate})...")
        history = []
        for epoch in range(epochs):
            loss = self.backpropagate(X, y, learning_rate)
            history.append(loss)
            
            # Log do progresso
            if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
                print(f"  Época {epoch + 1:>{len(str(epochs))}}/{epochs}, Erro (MSE): {loss:.8f}")
                
        print("Treinamento concluído.")
        return history

    def predict(self, X):
        """Faz previsões com a rede treinada (apenas feedforward)"""
        y_pred, _, _ = self.feedforward(X)
        return y_pred

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