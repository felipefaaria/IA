import pandas as pd
import numpy as np
import json
from collections import Counter

# Função para carregar e preparar os dados
def prepare_data(file_path):
    """
    Carrega o dataset, trata valores ausentes e discretiza atributos contínuos.
    """
    # Carregando dados
    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1')
    
    # Normalizando nomes de colunas
    df.columns = [col.strip() for col in df.columns]
    
    # Tratamento de Missing Values
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    most_frequent_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(most_frequent_embarked)
    
    # Removendo colunas que não serão usadas
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    # Discretização para o ID3
    # Discretizando 'Age' em 8 faixas
    bins_age = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels_age = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['Age'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age)
    
    # Discretizando 'Fare'
    df['Fare'] = pd.qcut(df['Fare'], q=4, labels=['low', 'medium', 'high', 'very_high'])
    
    # Discretizando 'SibSp' e 'Parch'
    df['SibSp'] = df['SibSp'].apply(lambda x: '0' if x == 0 else '1+')
    df['Parch'] = df['Parch'].apply(lambda x: '0' if x == 0 else '1+')
    
    return df

# Funções de utilidade
def entropy(target_col):
    """
    Calcula a entropia de uma coluna.
    """
    elements, counts = np.unique(target_col, return_counts=True)
    total_elements = len(target_col)
    entropy_val = np.sum([(-counts[i]/total_elements) * np.log2(counts[i]/total_elements) for i in range(len(elements)) if counts[i] > 0])
    return entropy_val

def information_gain(data, split_attribute, target_name="Survived"):
    """
    Calcula o ganho de informação.
    """
    total_entropy = entropy(data[target_name])
    
    values, counts = np.unique(data[split_attribute], return_counts=True)
    
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute]==values[i]).dropna()[target_name]) for i in range(len(values))])
    
    gain = total_entropy - weighted_entropy
    return gain

# Algoritmo ID3
def id3(data, original_data, features, target_attribute_name="Survived", parent_node_class=None, max_depth=None, current_depth=0):
    """
    Algoritmo ID3 para construir a árvore de decisão.
    """
    # Condição de parada para o limite de profundidade
    if max_depth is not None and current_depth >= max_depth:
        return int(np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])])
        
    # Condição de parada 1: Se o nó for puro, retorne a classe majoritária.
    unique_classes = np.unique(data[target_attribute_name])
    if len(unique_classes) <= 1:
        return int(unique_classes[0])
    
    # Condição de parada 2: Se não houver atributos, retorne a classe majoritária do conjunto original.
    elif len(features) == 0:
        return int(np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])])
    
    # Caso recursivo: Construa a árvore
    else:
        # Encontre a classe majoritária do nó para ser o nó pai (em caso de nós folha)
        parent_node_class = int(np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])])
        
        # Selecione o atributo com o maior ganho de informação
        item_gains = [information_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_gains)
        best_feature = features[best_feature_index]
        
        # Crie a estrutura da árvore com o melhor atributo
        # Converte a chave para string para garantir a compatibilidade JSON
        tree = {str(best_feature): {}}
        
        # Remova o melhor atributo da lista de features
        features = [i for i in features if i != best_feature]
        
        # Construa os nós filhos
        for value in np.unique(data[best_feature]):
            # Converte o valor para o tipo nativo do Python
            if isinstance(value, np.integer):
                key_value = int(value)
            else:
                key_value = str(value)

            sub_data = data[data[best_feature] == value]
            
            # Condição de parada 3: Se o sub-nó estiver vazio, retorne a classe majoritária do nó pai.
            if len(sub_data) == 0:
                tree[str(best_feature)][key_value] = parent_node_class
            else:
                subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class, max_depth, current_depth + 1)
                tree[str(best_feature)][key_value] = subtree
            
        return tree


if __name__ == '__main__':
    file_path = 'train.csv'
    
    try:
        # Preparar e discretizar os dados
        processed_df = prepare_data(file_path)
        
        # Separar os dados em features e target
        features_list = processed_df.drop('Survived', axis=1).columns.tolist()
        
        # Construa a árvore de decisão com profundidade limitada a 3
        titanic_tree_limited = id3(processed_df, processed_df, features_list, max_depth=3)
        
        print("Árvore de Decisão ID3 (Estrutura de Dicionário, Profundidade Limitada a 3):")
        # Imprime a árvore de forma legível
        print(json.dumps(titanic_tree_limited, indent=4))
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado. Por favor, certifique-se de que o arquivo está no mesmo diretório do script.")