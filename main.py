import pandas as pd
import plotly.graph_objects as go
import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import numpy as np


nltk.download('punkt_tab')
nltk.download('stopwords')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


def plot_publications_per_year(df):
    return df.groupby('PY').size()


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
    return df['T2'].str.lower().value_counts().head(top_n)


def top_countries(df, col='Country', top_n=15):

    df_countries_exploded = df[col].str.split(';').explode().str.strip()

    return df_countries_exploded.value_counts().head(top_n)


def most_cited_papers(df, N=10):

    df_ordenado = df.sort_values(by='Num_Citacoes', ascending=False)

    return df_ordenado.head(N)


def co_citation_analysis(df):
    co_citations = df.groupby('TI')['citation_count'].apply(list)
    # Implementar lógica de co-citação e criar gráfico de rede
    pass


def most_frequent_keywords(df, top_n=10):
    keywords = df['KW'].str.lower().str.split(';').explode().str.strip()
    print(f"Quantidade de palavras-chave: {len(keywords)}", file=f)
    common_keywords = keywords.value_counts().head(top_n)
    return common_keywords


import itertools
from collections import Counter
import networkx as nx
def keywords_cooccurrence(df):
    def ajustar_rotulo(rotulo):
        palavras = rotulo.split()
        if len(palavras) > 2:
            return '\n'.join(palavras)  # Inserir quebra de linha entre as palavras
        return rotulo

    df['KW_list'] = df['KW'].str.lower().str.replace(r'[^a-z; ]', ' ', regex=True).str.replace(r'  ', ' ', regex=True).str.split(';')
    df['KW_list'] = df['KW_list'].apply(
        lambda lista: [palavra.strip() for palavra in lista] if isinstance(lista, list) else lista)
    pares_coocorrentes = []

    for palavras in df['KW_list']:
        if isinstance(palavras, list):
            pares = list(itertools.combinations(sorted(set(palavras)), 2))
            pares_coocorrentes.extend(pares)

    contagem_coocorrencias  = Counter(pares_coocorrentes).most_common(100)

    grafo = nx.Graph()
    pesos_transformados = []

    for (palavra1, palavra2), peso in contagem_coocorrencias:
        peso_transformado = np.sqrt(peso)
        grafo.add_edge(palavra1, palavra2, weight=peso_transformado)

    grau_nos = dict(grafo.degree())
    node_size = [100 * grafo.degree(n) for n in grafo.nodes()]
    maior_no = max(grafo.degree, key=lambda x: x[1])[0]
    font_size_labels = {no: max(grau_nos[no] * 0.8, 10) for no in grafo.nodes()}

    edges = grafo.edges(data=True)
    colors = ["#3cbb75ff" for (u, v, d) in edges]

    shells = [[maior_no], list(set(grafo.nodes()) - {maior_no})]
    pos = nx.shell_layout(grafo, nlist=shells)
    nx.draw_networkx_nodes(grafo, pos, node_size=node_size, node_color="#55c667ff", alpha=0.8)
    nx.draw_networkx_edges(grafo, pos, width=[d['weight'] for (u, v, d) in edges], edge_color=colors, alpha=0.3)
    for no, (x, y) in pos.items():
        plt.text(x, y, s=ajustar_rotulo(no),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=font_size_labels[no])
    plt.axis('off')
    plt.show()

def analyze_freq_treemap(df, column='AB'):
    stop_words = set(stopwords.words('english'))

    df['AB_set_filtered'] = df[column].str.findall(r'\b[a-zA-Z]+\b')

    word_tokens = df['AB_set_filtered'].explode().to_list()

    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]

    df_all_words = pd.DataFrame(filtered_sentence, columns=['Palavra'])

    df_all_words_grouped = df_all_words.groupby('Palavra').agg({'Palavra': ['count']}).reset_index()

    df_all_words_grouped.columns = ['Palavra', 'count']

    df_all_words_grouped_top = df_all_words_grouped.sort_values('count', ascending=False).head(250)
    fig = px.treemap(
        df_all_words_grouped,
        path=[px.Constant('Coocorrência de palavras do resumo'), df_all_words_grouped_top['Palavra']],
        values=df_all_words_grouped_top['count'],
        title='Treemap Simples de Palavras e contagem'
    )
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.show()

    return df_all_words_grouped.sort_values('count', ascending=False).head(20)

def wordtree(df, coluna_texto='AB', termo_central='system'):

    de = []
    para = []

    for texto in df[coluna_texto].dropna().astype(str):
        palavras = re.split(r'\W+', texto.lower())
        try:
            index = palavras.index(termo_central.lower())

            if index > 2:
                de.append(f"-3. {palavras[index - 3]}")
                para.append(f"-2. {palavras[index - 2]}")

            if index > 1:
                de.append(f"-2. {palavras[index - 2]}")
                para.append(f"-1. {palavras[index - 1]}")

            if index > 0:
                de.append(f"-1. {palavras[index - 1]}" if index > 0 else "-1. ")
                para.append(f"+0. {palavras[index]}")

            if index < len(palavras) - 1:
                de.append(f"+0. {palavras[index]}")
                para.append(f"+1. {palavras[index + 1]}" if index < len(palavras) - 1 else "+1. ")

            if index < len(palavras) - 2:
                de.append(f"+1. {palavras[index + 1]}")
                para.append(f"+2. {palavras[index + 2]}")

            if index < len(palavras) - 3:
                de.append(f"+2. {palavras[index + 2]}")
                para.append(f"+3. {palavras[index + 3]}")

        except:
            continue

    df_transicoes = pd.DataFrame({"De": de, "Para": para})

    df_agrupado = df_transicoes.groupby(["De", "Para"], sort=False).size().reset_index(name='Contagem')

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
            color=[mcolors.to_hex(color) for color in sns.color_palette("crest", n_colors=2*len(sources))]
        )
    ))

    fig.update_layout(title_text=f"Sankey Chart para '{termo_central}'", font_family="Times New Roman", font_size=20)
    fig.show()


def run_bibliometric_analysis(df):
    #print(f"\nPublications per year:\n{plot_publications_per_year(df)}", file=f)
    #print(f"\nTop Authors:\n{top_authors(df).to_string()}", file=f)
    #print(f"\nTop Institutions:\n{top_institutions(df).to_string()}", file=f)
    #print(f"\nTop Journals:\n{top_journals(df).to_string()}", file=f)
    #print(f"\nTop Countries:\n{top_countries(df).to_string()}", file=f)
    #print(f"\nMost Cited Papers:\n{most_cited_papers(df)[['AU', 'TI', 'PY', 'T2', 'DO', 'Num_Citacoes']].to_string()}", file=f)
    #print(f"\nWord Frequency - Abstract:\n{analyze_freq_treemap(df, column='AB')}", file=f)

    #Figures
    keywords_cooccurrence(df)
    #wordtree(most_cited_papers(df, 30), 'AB', termo_central='energy')

if __name__ == "__main__":
    df = pd.read_csv('all_data_prepared.csv')

    f = open("output.txt", "a", encoding="utf-8")

    run_bibliometric_analysis(df)
