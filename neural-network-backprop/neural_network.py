import numpy as np

# =============================================================================
# --- Rede Neural com Backpropagation (implementada do zero) ---
# Módulo compartilhado pelos scripts xor.py e display.py.
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
    return np.mean((y_true - y_pred) ** 2)


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
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            # Biases inicializados com zeros
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def feedforward(self, X):
        """Propaga a entrada (X) pela rede (fase de inferência)"""
        activations = [X]  # Armazena as ativações (saídas) de cada camada
        inputs = []        # Armazena as entradas ponderadas (net) de cada camada

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
                sigmoid_derivative(inputs[i - 1])
            deltas.insert(0, delta_hidden)  # Adiciona o delta no início da lista

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
        # Evita divisão por zero no log de progresso quando epochs < 10
        log_step = max(1, epochs // 10)
        for epoch in range(epochs):
            loss = self.backpropagate(X, y, learning_rate)
            history.append(loss)

            # Log do progresso
            if (epoch + 1) % log_step == 0 or epoch == 0:
                print(f"  Época {epoch + 1:>{len(str(epochs))}}/{epochs}, Erro (MSE): {loss:.8f}")

        print("Treinamento concluído.")
        return history

    def predict(self, X):
        """Faz previsões com a rede treinada (apenas feedforward)"""
        y_pred, _, _ = self.feedforward(X)
        return y_pred
