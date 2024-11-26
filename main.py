# importar as bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_blobs, load_wine
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# carregar os datasets
def carregar_dataset(nome, tamanho=None):
    if nome == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        if tamanho:
            df = df.sample(n=tamanho, random_state=42)
        return df, data.target_names

    elif nome == "Blobs":
        X, y = make_blobs(n_samples=tamanho or 300, centers=3, random_state=42)
        df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        df['target'] = y
        return df, ["Cluster 0", "Cluster 1", "Cluster 2"]

    elif nome == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        if tamanho:
            df = df.sample(n=tamanho, random_state=42)
        return df, data.target_names

# treinar modelos
def treinar_modelo(df, algoritmo, parametros):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if algoritmo == "K-Means":
        n_clusters = parametros.get("n_clusters", 3)
        modelo = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = modelo.fit_predict(X)
        score = silhouette_score(X, clusters)
        return clusters, score, modelo

    elif algoritmo == "Árvore de Decisão":
        max_depth = parametros.get("max_depth", None)
        modelo = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        modelo.fit(X, y)
        previsoes = modelo.predict(X)
        score = accuracy_score(y, previsoes)
        return previsoes, score, modelo

    elif algoritmo == "Regressão Logística":
        modelo = LogisticRegression(random_state=42, max_iter=200)
        modelo.fit(X, y)
        previsoes = modelo.predict(X)
        score = accuracy_score(y, previsoes)
        return previsoes, score, modelo

    elif algoritmo == "SVM":
        modelo = SVC(kernel=parametros.get("kernel", "linear"), random_state=42)
        modelo.fit(X, y)
        previsoes = modelo.predict(X)
        score = accuracy_score(y, previsoes)
        return previsoes, score, modelo

    elif algoritmo == "KNN":
        n_neighbors = parametros.get("n_neighbors", 5)
        modelo = KNeighborsClassifier(n_neighbors=n_neighbors)
        modelo.fit(X, y)
        previsoes = modelo.predict(X)
        score = accuracy_score(y, previsoes)
        return previsoes, score, modelo

# visualizzr
def plotar_resultados(df, resultados, algoritmo, modelo):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    if algoritmo == "K-Means":
        plt.scatter(X[:, 0], X[:, 1], c=resultados, cmap="viridis", alpha=0.6)
        plt.scatter(modelo.cluster_centers_[:, 0], modelo.cluster_centers_[:, 1], s=50, c="red", marker="X", linewidths=0)
        plt.title("Clusters formados pelo K-Means")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        st.pyplot(plt.gcf())

    elif algoritmo == "Árvore de Decisão":
        plt.figure(figsize=(15, 10))
        plot_tree(modelo, feature_names=df.columns[:-1], class_names=[str(i) for i in set(y)], filled=True)
        plt.title("Árvore de Decisão")
        st.pyplot(plt.gcf())

    elif algoritmo in ["Regressão Logística", "SVM", "KNN"]:
        cm = confusion_matrix(y, resultados)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title("Matriz de Confusão")
        plt.xlabel("Previsões")
        plt.ylabel("Verdadeiro")
        st.pyplot(plt.gcf())

def descricao_algoritmo(algoritmo):
    descricoes = {
        "K-Means": "Um algoritmo de clustering que separa os dados em k grupos com base na similaridade.",
        "Árvore de Decisão": "Um modelo de classificação baseado em uma estrutura de árvore que toma decisões sequenciais.",
        "Regressão Logística": "Um modelo de classificação linear usado para prever categorias.",
        "SVM": "Algoritmo de classificação que encontra o hiperplano que melhor separa as classes.",
        "KNN": "Um classificador baseado em proximidade que considera os k vizinhos mais próximos."
    }
    return descricoes.get(algoritmo, "")

# interface com streamlit
st.title("Algoritmos Clássicos de Machine Learning")
st.sidebar.header("Configurações")

nome_dataset = st.sidebar.selectbox("Escolha o Dataset", ["Iris", "Blobs", "Wine"])
tamanho_dataset = st.sidebar.slider("Tamanho do Conjunto de Dados", min_value=30, max_value=150, value=100, step=10)
df, target_names = carregar_dataset(nome_dataset, tamanho_dataset)
st.write("### Visualização dos Dados")
st.write(df.head())
st.markdown("---")

algoritmo = st.sidebar.selectbox("Escolha o Algoritmo", ["K-Means", "Árvore de Decisão", "Regressão Logística", "SVM", "KNN"])

st.write(f"### Sobre o Algoritmo: {algoritmo}")
st.write(descricao_algoritmo(algoritmo))

parametros = {}
if algoritmo == "K-Means":
    parametros["n_clusters"] = st.sidebar.slider("Número de Clusters", min_value=2, max_value=10, value=3)
elif algoritmo == "Árvore de Decisão":
    parametros["max_depth"] = st.sidebar.slider("Profundidade Máxima", min_value=1, max_value=10, value=3)
elif algoritmo == "SVM":
    parametros["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
elif algoritmo == "KNN":
    parametros["n_neighbors"] = st.sidebar.slider("Número de Vizinhos", min_value=1, max_value=20, value=5)

if st.sidebar.button("Treinar Modelo"):
    resultados, score, modelo = treinar_modelo(df, algoritmo, parametros)
    st.write(f"### Desempenho do Modelo ({algoritmo})")
    st.write(f"Acurácia: {score:.2f}/1.00")

    st.write("### Resultados Visualizados")
    plotar_resultados(df, resultados, algoritmo, modelo)

st.markdown("---")
st.markdown("### Contato")
st.markdown("Meu GitHub: [PedroRebello1](https://github.com/PedroRebello1)")
st.markdown("Meu LinkedIn: [Pedro Rebello](https://www.linkedin.com/in/pedro-rebello-a43b562b7/)")
st.markdown("E-mail: [pedrorebellozf@gmail.com](mailto:pedrorebellozf@gmail.com)")
