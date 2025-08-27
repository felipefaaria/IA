import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    """
    Executa a análise completa da base de dados do Titanic.
    """
    # --- Passo 1: Carregamento dos Dados ---
    print("--- Carregando a base de dados do Titanic (train.csv) ---")
    try:
        titanic_df = pd.read_csv('./train.csv')
        print("DataFrame carregado com sucesso.")
    except FileNotFoundError:
        print("Erro: O arquivo 'train.csv' não foi encontrado. Por favor, certifique-se de que ele está no mesmo diretório.")
        return

    # --- Passo 2: Pré-processamento e Codificação dos Dados ---
    print("\n--- Pré-processamento dos dados ---")
    df = titanic_df.copy()

    # Lidando com valores ausentes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Remover colunas que não serão usadas para o modelo
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Codificar 'Sex' (binário) com LabelEncoder
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

    # Codificar 'Embarked' (não ordinal) com OneHotEncoder
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=True)

    print("\nDataFrame após o pré-processamento:")
    print(df.head())
    print("\nInformações finais:")
    df.info()

    # --- Passo 3: Divisão dos Dados e Otimização do Modelo ---
    print("\n--- Otimizando o modelo com GridSearchCV ---")
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Dividir em conjuntos de treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.20, random_state=42)

    # Definir os parâmetros para a busca
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2'],
    }

    modelo_dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=modelo_dt,
        param_grid=params,
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    
    grid_search.fit(X_treino, y_treino)

    print("\n--- Parâmetros do Modelo ---")
    print("Melhores hiperparâmetros encontrados:", grid_search.best_params_)
    print("Melhor acurácia (média da validação cruzada):", grid_search.best_score_)

    # --- Passo 4: Avaliação e Visualização do Modelo Final ---
    print("\n--- Avaliando o modelo final ---")
    melhor_modelo = grid_search.best_estimator_
    previsoes = melhor_modelo.predict(X_teste)

    print("\nAcurácia no conjunto de teste:", accuracy_score(y_teste, previsoes))
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, previsoes))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_teste, previsoes))

    # Visualizar a árvore de decisão
    print("\n--- Gerando a Árvore de Decisão ---")
    feature_names = X.columns.tolist()
    class_names = ['Não Sobreviveu', 'Sobreviveu']

    plt.figure(figsize=(25, 15))
    plot_tree(
        melhor_modelo,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        fontsize=10
    )
    plt.title('Árvore de Decisão - Padrão de Sobrevivência no Titanic')
    plt.savefig('titanic_decision_tree.png')
    print("Árvore de decisão salva como 'titanic_decision_tree.png'.")

if __name__ == "__main__":
    main()