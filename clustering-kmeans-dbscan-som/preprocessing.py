import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

sns.set(style="whitegrid")

def run_preprocessing():

    print("--- Etapa 1: Carregando Base de Dados ---")
    try:
        #Adicione o arquivo
        df = pd.read_csv('creditcard.csv')
        print(f"Dataset carregado com sucesso. Dimensões originais: {df.shape}")
    except FileNotFoundError:
        print("ERRO: Arquivo 'creditcard.csv' não encontrado. Verifique o caminho.")
        return

    print(f"Distribuição das Classes:\n{df['Class'].value_counts(normalize=True)}")
    
    print("\n--- Etapa 2: Valores Ausentes ---")
    missing = df.isnull().sum().max()
    print(f"Total de valores nulos encontrados: {missing}")
    if missing > 0:
        df.fillna(df.median(), inplace=True)
        print("Valores nulos preenchidos com a mediana.")
    else:
        print("Nenhum tratamento de nulos necessário.")

    print("\n--- Etapa 3: Remoção de Duplicatas ---")
    duplicates = df.duplicated().sum()
    print(f"Entradas duplicadas encontradas: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Duplicatas removidas. Novas dimensões: {df.shape}")

    print("\n--- Etapa 4: Análise de Outliers (Amount) ---")

    Q1 = df['Amount'].quantile(0.25)
    Q3 = df['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    outliers_count = ((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR))).sum()
    print(f"Outliers estatísticos detectados na coluna Amount: {outliers_count}")

    print("\n--- Etapa 5: Normalização (RobustScaler) ---")

    scaler = RobustScaler()
    

    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    

    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    print("Colunas 'Time' e 'Amount' normalizadas e originais removidas.")

    print("\n--- Etapa 6: Correlação ---")

    corr_target = df.corrwith(df['Class']).sort_values(ascending=False)
    print("Top 5 correlações positivas com 'Class':")
    print(corr_target.head(5))
    print("Top 5 correlações negativas com 'Class':")
    print(corr_target.tail(5))

    print("\n--- Etapa 7: Codificação ---")
    print("Dataset contém apenas variáveis numéricas (float64). Nenhuma codificação necessária.")

    print("\n--- Etapa 8: Divisão Treino/Teste Estratificada ---")
    X = df.drop('Class', axis=1)
    y = df['Class']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Shape Treino: {X_train.shape}, Shape Teste: {X_test.shape}")


    print("\n--- Etapa 9: Balanceamento (SMOTE) ---")
    print("Aplicando SMOTE apenas nos dados de treino...")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("Antes do SMOTE (y_train):")
    print(y_train.value_counts())
    print("Depois do SMOTE (y_train_resampled):")
    print(y_train_resampled.value_counts())

    print("\n=== PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO ===")
    print("Retornando: X_train_balanced, X_test, y_train_balanced, y_test")
    
    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    X_train_bal, X_test, y_train_bal, y_test = run_preprocessing()