# Código do algoritmo CART (Corrigido)

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
    
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    return df

def gini_index(target_col):
    """
    Calcula o índice de Gini de uma coluna.
    """
    elements, counts = np.unique(target_col, return_counts=True)
    total_elements = len(target_col)
    
    p_sq = np.sum([(counts[i]/total_elements)**2 for i in range(len(elements))])
    gini = 1 - p_sq
    return gini

def weighted_gini(subsets, target_name):
    """
    Calcula o Gini ponderado para uma divisão.
    """
    total = sum([len(subset) for subset in subsets])
    weighted_gini_val = sum([(len(subset)/total) * gini_index(subset[target_name]) for subset in subsets if len(subset) > 0])
    return weighted_gini_val

def find_best_split_for_continuous_cart(data, attribute, target_name):
    """
    Encontra o melhor limiar para um atributo contínuo usando o critério de Gini.
    """
    unique_values = data[attribute].unique()
    unique_values.sort()
    
    best_threshold = None
    min_weighted_gini = 1
    
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i+1]) / 2
        
        subset_le = data[data[attribute] <= threshold]
        subset_gt = data[data[attribute] > threshold]
        
        if len(subset_le) == 0 or len(subset_gt) == 0:
            continue
        
        current_weighted_gini = weighted_gini([subset_le, subset_gt], target_name)
        
        if current_weighted_gini < min_weighted_gini:
            min_weighted_gini = current_weighted_gini
            best_threshold = threshold
            
    gain = gini_index(data[target_name]) - min_weighted_gini if min_weighted_gini < 1 else 0
    return gain, best_threshold

def find_best_split_for_categorical_cart(data, attribute, target_name):
    """
    Encontra a melhor divisão binária para um atributo categórico usando o critério de Gini.
    """
    values = np.unique(data[attribute])
    if len(values) <= 1:
        return -1, None
    
    min_weighted_gini = 1
    best_split_set = None
    
    from itertools import combinations
    for k in range(1, len(values) // 2 + 1):
        for combo in combinations(values, k):
            set1 = list(combo)
            set2 = [v for v in values if v not in set1]
            
            subset1 = data[data[attribute].isin(set1)]
            subset2 = data[~data[attribute].isin(set1)]
            
            if len(subset1) == 0 or len(subset2) == 0:
                continue

            current_weighted_gini = weighted_gini([subset1, subset2], target_name)
            
            if current_weighted_gini < min_weighted_gini:
                min_weighted_gini = current_weighted_gini
                best_split_set = set1
    
    gain = gini_index(data[target_name]) - min_weighted_gini if min_weighted_gini < 1 else 0
    return gain, best_split_set

def cart_tree(data, original_data, features, target_attribute_name="Survived", parent_node_class=None):
    if len(data) == 0:
        return parent_node_class
        
    unique_classes = np.unique(data[target_attribute_name])
    if len(unique_classes) <= 1:
        return int(unique_classes[0])
    
    elif len(features) == 0:
        return int(np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])])
    
    else:
        parent_node_class = int(np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])])
        
        max_gini_gain = -1
        best_feature = None
        split_condition = None
        
        for feature in features:
            if data[feature].dtype in [np.float64, np.int64]:
                gain, threshold = find_best_split_for_continuous_cart(data, feature, target_attribute_name)
                if gain > max_gini_gain:
                    max_gini_gain = gain
                    best_feature = feature
                    split_condition = threshold
            else:
                gain, split_set = find_best_split_for_categorical_cart(data, feature, target_attribute_name)
                if gain > max_gini_gain:
                    max_gini_gain = gain
                    best_feature = feature
                    split_condition = split_set

        if best_feature is None or max_gini_gain == 0:
            return parent_node_class

        tree = {str(best_feature): {}}
        remaining_features = [f for f in features if f != best_feature]

        if isinstance(split_condition, list):
            subset1 = data[data[best_feature].isin(split_condition)]
            subset2 = data[~data[best_feature].isin(split_condition)]
            
            tree[str(best_feature)][f"isin({split_condition})"] = cart_tree(subset1, original_data, remaining_features, target_attribute_name, parent_node_class)
            tree[str(best_feature)][f"not isin({split_condition})"] = cart_tree(subset2, original_data, remaining_features, target_attribute_name, parent_node_class)
        else:
            subset1 = data[data[best_feature] <= split_condition]
            subset2 = data[data[best_feature] > split_condition]
            
            tree[str(best_feature)][f"<= {split_condition:.2f}"] = cart_tree(subset1, original_data, remaining_features, target_attribute_name, parent_node_class)
            tree[str(best_feature)][f"> {split_condition:.2f}"] = cart_tree(subset2, original_data, remaining_features, target_attribute_name, parent_node_class)

        return tree


if __name__ == '__main__':
    file_path = 'train.csv'
    
    try:
        processed_df = prepare_data(file_path)
        
        features_list = processed_df.drop('Survived', axis=1).columns.tolist()
        
        titanic_tree = cart_tree(processed_df, processed_df, features_list)
        
        print("Árvore de Decisão CART (Estrutura de Dicionário):")
        print(json.dumps(titanic_tree, indent=4))
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado. Por favor, certifique-se de que o arquivo está no mesmo diretório do script.")