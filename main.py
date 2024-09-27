import bibtexparser
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations


# 5.1.1 Evolução Temporal: Número de publicações por ano
def plot_publications_per_year(df):
    yearly_publications = df.groupby('Year').size()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=yearly_publications.index, y=yearly_publications.values, color='skyblue')
    plt.title('Distribuição de Publicações por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Número de Publicações')
    plt.savefig('publications_per_year.png', format='png')
    plt.close()


# 5.1.2 Principais Autores: Lista dos autores com maior número de publicações e citações
def top_authors(df, top_n=10):
    # Verifique se a coluna 'author' está presente
    if 'Author Full Names' not in df.columns:
        raise KeyError("A coluna 'author' não foi encontrada no DataFrame.")

    all_authors = []

    for authors in df['Author Full Names']:
        all_authors.extend([author.strip() for author in authors.split(';')])

    author_counts = pd.Series(all_authors).value_counts().head(top_n)

    top_authors_df = pd.DataFrame({'Publicações': author_counts})

    return top_authors_df


# 5.1.3 Instituições: Principais contribuintes
def top_institutions(df, top_n=10):
    affiliations = df['Affiliations'].str.split(';').explode()
    affiliations = affiliations.str.split(',')
    return affiliations.value_counts().head(top_n)


# 5.1.4 Periódicos e Conferências: Veículos de publicação mais relevantes
def top_journals(df, top_n=10):
    return df['Source Title'].value_counts().head(top_n)


# 5.2 Análise de Citação

# 5.2.1 Artigos Mais Citados
def most_cited_papers(df, top_n=10):
    most_cited = df.sort_values('Cited', ascending=False).head(top_n)
    return most_cited[['Title', 'Author Full Names', 'Cited']]


# 5.2.2 Análise de Co-citação
def co_citation_analysis(df):
    co_citations = df.groupby('Title')['citation_count'].apply(list)
    # Implementar lógica de co-citação e criar gráfico de rede
    pass


# 5.3 Análise de Redes de Colaboração
def collaboration_network(df, coluna_autores='Author Full Names'):
    G = nx.Graph()

    for autores in df[coluna_autores].dropna():
        lista_autores = [autor.strip() for autor in autores.split(';')]

        for autor1, autor2 in combinations(lista_autores, 2):
            if G.has_edge(autor1, autor2):
                G[autor1][autor2]['weight'] += 1
            else:
                G.add_edge(autor1, autor2, weight=1)

    print(f"Número de nós (autores): {G.number_of_nodes()}")
    print(f"Número de arestas (colaborações): {G.number_of_edges()}")

    centralidade_grau = nx.degree_centrality(G)

    top_autores = sorted(centralidade_grau.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 autores mais colaborativos:")
    for autor, centralidade in top_autores:
        print(f"{autor}: {centralidade:.4f}")

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5, seed=42)  # Layout para a visualização da rede
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=[d['weight'] for (u, v, d) in G.edges(data=True)], alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    plt.title("Rede de Colaboração entre Autores", size=15)
    plt.savefig('collaboration_network.png', format='png')
    plt.close()

# 5.4.1 Palavras-chave Mais Frequentes
def most_frequent_keywords(df, top_n=10):
    keywords = df['Keywords'].str.lower().str.split(';').explode()
    common_keywords = keywords.value_counts().head(top_n)
    return common_keywords

# 5.4.2 Análise de Coocorrência
def analyze_cooccurrence(df, text_column='Abstract'):
    def preprocess_text(text):
        return text.lower()

    vectorizer = CountVectorizer(stop_words='english', tokenizer=lambda text: preprocess_text(text).split())

    X = vectorizer.fit_transform(df[text_column])
    df_vectorized = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    cooccurrence_matrix = df_vectorized.T.dot(df_vectorized)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence_matrix, annot=True, cmap='YlGnBu', xticklabels=cooccurrence_matrix.columns,
                yticklabels=cooccurrence_matrix.columns)
    plt.title('Matriz de Coocorrência de Palavras')
    plt.show()

    G = nx.from_numpy_matrix(cooccurrence_matrix.values)
    labels = {i: term for i, term in enumerate(cooccurrence_matrix.columns)}

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title('Rede de Coocorrência de Palavras')
    plt.savefig('cooccurrence_network.png')
    plt.close()


# Função principal para rodar todas as análises
def run_bibliometric_analysis(df):
    plot_publications_per_year(df)

    print(f"\nTop Authors:\n{top_authors(df)}")

    print(f"\nTop Institutions:\n{top_institutions(df)}")

    print(f"\nTop Journals:\n{top_journals(df)}")

    print(f"\nMost Cited Papers:\n{most_cited_papers(df)}")

    collaboration_network(df)

    print(f"\nMost Frequent Keywords:\n{most_frequent_keywords(df)}")

    #analyze_cooccurrence(df, 'abstract')


if __name__ == "__main__":
    df = pd.read_csv('all_data_filtered_normalized.csv')

    print(df.columns)

    run_bibliometric_analysis(df)
