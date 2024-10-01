import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations
from wordcloud import WordCloud
import plotly.graph_objects as go
import re


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


def plot_publications_per_year(df):
    yearly_publications = df.groupby('PY').size()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=yearly_publications.index, y=yearly_publications.values, color='skyblue')
    plt.title('Distribuição de Publicações por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Número de Publicações')
    plt.savefig('publications_per_year.png', format='png')
    plt.close()


def top_authors(df, top_n=10):
    all_authors = []

    for authors in df['AU']:
        all_authors.extend([author.strip() for author in authors.split(';')])

    author_counts = pd.Series(all_authors).value_counts().head(top_n)

    top_authors_df = pd.DataFrame({'Publicações': author_counts})

    return top_authors_df


def top_institutions(df, top_n=10):
    affiliations = df['Institution'].str.split(';').explode()
    return affiliations.value_counts().head(top_n)


def top_journals(df, top_n=10):
    return df['T2'].value_counts().head(top_n)


def top_countries(df, col='Country', top_n=15):

    df_countries_exploded = df[col].str.split(';').explode().str.strip()

    return df_countries_exploded.value_counts().head(top_n)

def extrair_numero_citacoes(citacao):
    match = re.search(r'(?:Cited By|Times Cited).*?(\d+)', citacao)
    if match:
        return int(match.group(1))
    return 0

def most_cited_papers(df, coluna_citacoes = 'N1', N=10):
    df['Num_Citacoes'] = df[coluna_citacoes].apply(extrair_numero_citacoes)

    df_ordenado = df.sort_values(by='Num_Citacoes', ascending=False)

    return df_ordenado.head(N)


def co_citation_analysis(df):
    co_citations = df.groupby('TI')['citation_count'].apply(list)
    # Implementar lógica de co-citação e criar gráfico de rede
    pass


def most_frequent_keywords(df, top_n=10):
    keywords = df['KW'].str.lower().str.split(';').explode().str.strip()
    common_keywords = keywords.value_counts().head(top_n)
    return common_keywords


def analyze_cooccurrence(df, text_column='KW'):
    # Transformar a coluna de palavras-chave em minúsculas, separar por ponto e vírgula e remover espaços
    keywords_series = df[text_column].fillna('').str.lower().str.split(';').apply(lambda x: [k.strip() for k in x if isinstance(k, str)])

    # Contar as co-ocorrências de pares de palavras-chave
    coocorrencias = Counter()
    for keywords in keywords_series:
        if len(keywords) > 1:  # Se houver mais de uma palavra-chave, gerar as combinações
            coocorrencias.update(combinations(sorted(keywords), 2))

    # Criar o grafo de co-ocorrência
    G = nx.Graph()

    # Adicionar arestas com a frequência de co-ocorrência
    for (keyword1, keyword2), freq in coocorrencias.items():
        G.add_edge(keyword1, keyword2, weight=freq)

    # Plotar o grafo
    plt.figure(figsize=(12, 12))

    # Definir o layout
    pos = nx.spring_layout(G, k=0.5)  # k ajusta o espaçamento entre os nós

    # Desenhar os nós e arestas
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] * 0.1 for u, v in G.edges()])

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title('Rede de Coocorrência de Palavras-Chave')
    plt.savefig('cooccurrence_network.png')
    plt.close()


def wordcloud(df, column='AB'):
    # Concatenar todos os textos do Abstract em uma única string
    texto = ' '.join(df[column].dropna().astype(str))

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis',
                          max_words=200, contour_color='steelblue').generate(texto)
    wordcloud.to_file('wordcloud.png')


def wordtree(df, coluna_texto='AB', termo_central='system'):

    de = []
    para = []

    for texto in df[coluna_texto].dropna().astype(str):
        palavras = re.split(r'\W+', texto.lower())
        try:
            index = palavras.index(termo_central.lower())

            if index > 1:
                de.append(f"-2. {palavras[index - 2]}")
                para.append(f"-1. {palavras[index - 1]}")

            de.append(f"-1. {palavras[index - 1]}" if index > 0 else "-1. ")
            para.append(f"+0. ")

            de.append(f"+0. ")
            para.append(f"+1. {palavras[index + 1]}" if index < len(palavras) - 1 else "+1. ")

            if index < len(palavras) - 2:
                de.append(f"+1. {palavras[index + 1]}")
                para.append(f"+2. {palavras[index + 2]}")

        except:
            continue

    df_transicoes = pd.DataFrame({"De": de, "Para": para})

    df_agrupado = df_transicoes.groupby(["De", "Para"]).size().reset_index(name='Contagem')

    labels = list(pd.unique(df_agrupado[['De', 'Para']].values.ravel('K')))

    sources = df_agrupado['De'].apply(lambda x: labels.index(x)).tolist()
    targets = df_agrupado['Para'].apply(lambda x: labels.index(x)).tolist()
    values = df_agrupado['Contagem'].tolist()

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=1,
            line=dict(color="rgba(190, 190, 190, 1)", width=0.5),
            label=[label[3:] for label in labels],
            color="rgba(0, 0, 0, 0)"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(0, 0, 0, 0.2)'
        )
    ))

    # Adicionar título e mostrar o gráfico
    fig.update_layout(title_text=f"Sankey Chart para '{termo_central}'", font_family="Times New Roman", font_size=20)
    fig.show()


def run_bibliometric_analysis(df):
    plot_publications_per_year(df)

    print(f"\nTop Authors:\n{top_authors(df).to_string()}", file=f)

    print(f"\nTop Institutions:\n{top_institutions(df).to_string()}", file=f)

    print(f"\nTop Journals:\n{top_journals(df).to_string()}", file=f)

    print(f"\nTop Journals:\n{top_countries(df).to_string()}", file=f)

    print(f"\nMost Cited Papers:\n{most_cited_papers(df)[['AU', 'TI', 'PY', 'T2', 'DO', 'Num_Citacoes']].to_string()}", file=f)

    print(f"\nMost Frequent Keywords:\n{most_frequent_keywords(df).to_string()}", file=f)

    wordcloud(df)

    wordtree(most_cited_papers(df, 'N1', 100), 'AB', termo_central='model')

if __name__ == "__main__":
    df = pd.read_csv('all_data_prepared.csv')

    f = open("output.txt", "a", encoding="utf-8")

    run_bibliometric_analysis(df)
