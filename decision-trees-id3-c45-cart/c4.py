import pandas as pd
import numpy as np
import json
from collections import Counter
from sklearn.model_selection import train_test_split

def prepare_data(file_path):
    """
    Carrega o dataset, trata valores ausentes e retorna um dataframe sem discretização.
    """
    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1')
    
    df.columns = [col.strip() for col in df.columns]
    
    # Tratamento de Missing Values
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Removendo colunas irrelevantes
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    return df

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    total_elements = len(target_col)
    entropy_val = np.sum([(-counts[i]/total_elements) * np.log2(counts[i]/total_elements) for i in range(len(elements)) if counts[i] > 0])
    return entropy_val

def information_gain(data, split_attribute, target_name="Survived"):
    total_entropy = entropy(data[target_name])
    
    values, counts = np.unique(data[split_attribute], return_counts=True)
    
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute]==values[i]).dropna()[target_name]) for i in range(len(values))])
    
    gain = total_entropy - weighted_entropy
    return gain

def split_info(data, split_attribute):
    values, counts = np.unique(data[split_attribute], return_counts=True)
    total = len(data)
    split_info_val = np.sum([-(counts[i]/total) * np.log2(counts[i]/total) for i in range(len(values)) if counts[i] > 0])
    return split_info_val

def gain_ratio(data, split_attribute, target_name="Survived"):
    gain = information_gain(data, split_attribute, target_name)
    split_info_val = split_info(data, split_attribute)
    if split_info_val == 0:
        return 0
    return gain / split_info_val

def find_best_split_for_continuous(data, attribute, target_name):
    unique_values = data[attribute].unique()
    unique_values.sort()
    
    best_threshold = None
    max_gain_ratio = -1
    
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i+1]) / 2
        
        subset_le = data[data[attribute] <= threshold]
        subset_gt = data[data[attribute] > threshold]
        
        if len(subset_le) == 0 or len(subset_gt) == 0:
            continue
        
        data_split = data.copy()
        data_split['split_attribute'] = data_split[attribute].apply(lambda x: 'le' if x <= threshold else 'gt')
        
        current_gain_ratio = gain_ratio(data_split, 'split_attribute', target_name)
        
        if current_gain_ratio is not None and current_gain_ratio > max_gain_ratio:
            max_gain_ratio = current_gain_ratio
            best_threshold = threshold
            
    return max_gain_ratio, best_threshold

def c45_tree(data, original_data, features, target_attribute_name="Survived", parent_node_class=None, max_depth=None, current_depth=0):
    # Condição de parada para o limite de profundidade
    if max_depth is not None and current_depth >= max_depth:
        return int(np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])])
        
    if len(np.unique(data[target_attribute_name])) <= 1:
        return int(np.unique(data[target_attribute_name])[0])
    
    elif len(features) == 0:
        return int(np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])])
    
    else:
        parent_node_class = int(np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])])
        
        best_gain_ratio = -1
        best_feature = None
        best_threshold = None
        
        categorical_features = [f for f in features if data[f].dtype not in [np.float64, np.int64]]
        continuous_features = [f for f in features if data[f].dtype in [np.float64, np.int64]]

        for feature in categorical_features:
            current_gain_ratio = gain_ratio(data, feature, target_attribute_name)
            if current_gain_ratio > best_gain_ratio:
                best_gain_ratio = current_gain_ratio
                best_feature = feature
                best_threshold = None
        
        for feature in continuous_features:
            current_gain_ratio, threshold = find_best_split_for_continuous(data, feature, target_attribute_name)
            if current_gain_ratio is not None and current_gain_ratio > best_gain_ratio:
                best_gain_ratio = current_gain_ratio
                best_feature = feature
                best_threshold = threshold
        
        if best_feature is None:
            return parent_node_class

        tree = {str(best_feature): {}}
        remaining_features = [f for f in features if f != best_feature]

        if best_threshold is not None:
            left_subset = data[data[best_feature] <= best_threshold]
            right_subset = data[data[best_feature] > best_threshold]

            tree[str(best_feature)][f"<= {best_threshold:.2f}"] = c45_tree(left_subset, original_data, remaining_features, target_attribute_name, parent_node_class, max_depth, current_depth + 1)
            tree[str(best_feature)][f"> {best_threshold:.2f}"] = c45_tree(right_subset, original_data, remaining_features, target_attribute_name, parent_node_class, max_depth, current_depth + 1)
        else:
            for value in np.unique(data[best_feature]):
                sub_data = data[data[best_feature] == value]
                if len(sub_data) == 0:
                    tree[str(best_feature)][str(value)] = parent_node_class
                else:
                    subtree = c45_tree(sub_data, original_data, remaining_features, target_attribute_name, parent_node_class, max_depth, current_depth + 1)
                    tree[str(best_feature)][str(value)] = subtree

        return tree

if __name__ == '__main__':
    file_path = 'train.csv'
    
    try:
        processed_df = prepare_data(file_path)
        
        features_list = processed_df.drop('Survived', axis=1).columns.tolist()
        
        # Gera a árvore com profundidade limitada a 3
        titanic_tree_limited = c45_tree(processed_df, processed_df, features_list, max_depth=3)
        
        print("Árvore de Decisão C4.5 (Profundidade Limitada a 3):")
        print(json.dumps(titanic_tree_limited, indent=4))
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado. Por favor, certifique-se de que o arquivo está no mesmo diretório do script.")