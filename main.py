import bibtexparser
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def standardize_columns(df):
    df.columns = [col.lower() for col in df.columns]
    return df


# 5.1.1 Evolução Temporal: Número de publicações por ano
def plot_publications_per_year(df):
    yearly_publications = df.groupby('year').size()
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
    if 'author full names' not in df.columns:
        raise KeyError("A coluna 'author' não foi encontrada no DataFrame.")

    all_authors = []

    for authors in df['author full names']:
        all_authors.extend([author.strip() for author in authors.split(';')])

    author_counts = pd.Series(all_authors).value_counts().head(top_n)

    top_authors_df = pd.DataFrame({'Publicações': author_counts})

    return top_authors_df


# 5.1.3 Instituições: Principais contribuintes
def top_institutions(df, top_n=10):
    affiliations = df['affiliations'].str.split(';').explode()
    affiliations = affiliations.str.split(',')
    return affiliations.value_counts().head(top_n)


# 5.1.4 Periódicos e Conferências: Veículos de publicação mais relevantes
def top_journals(df, top_n=10):
    return df['source title'].value_counts().head(top_n)


# 5.2 Análise de Citação

# 5.2.1 Artigos Mais Citados
def most_cited_papers(df, top_n=10):
    most_cited = df.sort_values('cited', ascending=False).head(top_n)
    return most_cited[['title', 'authors', 'cited']]


# 5.2.2 Análise de Co-citação
def co_citation_analysis(df):
    co_citations = df.groupby('title')['citation_count'].apply(list)
    # Implementar lógica de co-citação e criar gráfico de rede
    pass


# 5.3 Análise de Redes de Colaboração


# 5.4.1 Palavras-chave Mais Frequentes
def most_frequent_keywords(df, top_n=10):
    keywords = df['keywords'].str.lower().str.split(';').explode()
    common_keywords = keywords.value_counts().head(top_n)
    return common_keywords

# 5.4.2 Análise de Coocorrência
def analyze_cooccurrence(df, text_column='abstract'):
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

    print(f"\nTop Institutions:\n{top_authors(df)}")

    print(f"\nTop Institutions:\n{top_institutions(df)}")

    print(f"\nTop Journals:\n{top_journals(df)}")

    print(f"\nMost Cited Papers:\n{most_cited_papers(df)}")

    #collaboration_network(df)

    print(f"\nMost Frequent Keywords:\n{most_frequent_keywords(df)}")

    #analyze_cooccurrence(df, 'abstract')


def bib_to_dataframe(bib_file):
    try:
        with open(bib_file, 'r', encoding='utf-8') as file:
            bib_data = file.read()
    except UnicodeDecodeError:
        with open(bib_file, 'r', encoding='latin1') as file:
            bib_data = file.read()

    bib_database = bibtexparser.bparser.BibTexParser().parse(bib_data)
    entries = bib_database.entries
    df = pd.DataFrame(entries)

    return df

if __name__ == "__main__":
    scopus_df = pd.read_csv('scopus.csv')
    webofscience_df = pd.read_csv('webofscience.csv')

    scopus_df = standardize_columns(scopus_df)
    #print(scopus_df.columns)
    webofscience_df = standardize_columns(webofscience_df)
    #print(webofscience_df.columns)
    scopus_columns = set(scopus_df.columns)
    webofscience_columns = set(webofscience_df.columns)

    common_columns = scopus_columns.intersection(webofscience_columns)

    scopus_df_filtered = scopus_df[list(common_columns)]
    webofscience_df_filtered = webofscience_df[list(common_columns)]

    df = pd.concat([scopus_df_filtered, webofscience_df_filtered], ignore_index=True)

    print(df.columns)

    run_bibliometric_analysis(df)
