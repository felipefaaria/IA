# Trabalhos de Inteligência Artificial

Coletânea de implementações da disciplina de IA. Cada pasta é um projeto
independente com seu próprio `requirements.txt`.

## Projetos

| Pasta                                                           | Descrição                                                                                      |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [`decision-tree-sklearn/`](decision-tree-sklearn)               | Árvore de decisão com scikit-learn nos datasets Restaurante (notebook) e Titanic (script).     |
| [`decision-trees-id3-c45-cart/`](decision-trees-id3-c45-cart)   | Árvores de decisão implementadas do zero: ID3, C4.5 e CART, aplicadas ao Titanic.              |
| [`neural-network-backprop/`](neural-network-backprop)           | Rede neural com backpropagation implementada do zero (problemas XOR e display de 7 segmentos). |
| [`clustering-kmeans-dbscan-som/`](clustering-kmeans-dbscan-som) | Pré-processamento e clustering (K-Means, DBSCAN, SOM) na base de fraude de cartão de crédito.  |
| [`8puzzle/`](8puzzle)                                           | Resolvedor visual do 8-puzzle (BFS, Gulosa e A\* com várias heurísticas) em pygame.            |
| [`MazeSolver/`](MazeSolver)                                     | Resolvedor visual de labirinto (BFS, Gulosa e A\*) em pygame.                                  |

## Como executar

Cada projeto tem suas dependências listadas em `requirements.txt`:

```bash
cd <pasta-do-projeto>
pip install -r requirements.txt
python <script>.py
```

> **Nota:** os projetos `clustering-kmeans-dbscan-som` esperam o arquivo
> `creditcard.csv` (base de fraude de cartão do Kaggle) na própria pasta.
