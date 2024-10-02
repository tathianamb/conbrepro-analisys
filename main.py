import pandas as pd
from wordcloud import WordCloud
import plotly.graph_objects as go
import re
import matplotlib.colors as mcolors
import seaborn as sns


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
    return df['T2'].value_counts().head(top_n)


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
    print(f"Quantidade de palavras-chave: {len(keywords)}")
    common_keywords = keywords.value_counts().head(top_n)
    return common_keywords


def analyze_cooccurrence(df, column='AB'):
    df['conjuntos_palavras'] = df['AB'].apply(lambda x: set(x.split()))

    all_words = ' '.join([' '.join(conjunto) for conjunto in df['conjuntos_palavras']])

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis',
                          max_words=200, contour_color='steelblue').generate(all_words)

    list_keys = list(wordcloud.words_.keys())

    df_all_words = pd.DataFrame(all_words.split(), columns=['Palavra'])

    df_all_words_grouped = df_all_words.groupby('Palavra').count().reset_index()

    df_filtered = df_all_words_grouped[df_all_words_grouped['Palavra'].isin(list_keys)]

    wordcloud.to_file('wordcloud_analyze_cooccurrence.png')

    return df_filtered.sort_values('Palavra', ascending=False).head(20)

def analyze_freq(df, column='AB'):

    all_words = ' '.join(df[column].dropna().astype(str))

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis',
                          max_words=200, contour_color='steelblue').generate(all_words)

    list_keys = list(wordcloud.words_.keys())

    df_all_words = pd.DataFrame(all_words.split(), columns=['Palavra'])

    df_all_words_grouped = df_all_words.groupby('Palavra').count().reset_index()

    df_filtered = df_all_words_grouped[df_all_words_grouped['Palavra'].isin(list_keys)]

    wordcloud.to_file('wordcloud_analyze_freq.png')

    return df_filtered.sort_values('Palavra', ascending=False).head(20)

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
                para.append(f"+0. ")

            if index < len(palavras) - 1:
                de.append(f"+0. ")
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
    print(f"\nPublications per year:\n{plot_publications_per_year(df)}", file=f)

    print(f"\nTop Authors:\n{top_authors(df).to_string()}", file=f)

    print(f"\nTop Institutions:\n{top_institutions(df).to_string()}", file=f)

    print(f"\nTop Journals:\n{top_journals(df).to_string()}", file=f)

    print(f"\nTop Countries:\n{top_countries(df).to_string()}", file=f)

    print(f"\nMost Cited Papers:\n{most_cited_papers(df)[['AU', 'TI', 'PY', 'T2', 'DO', 'Num_Citacoes']].to_string()}", file=f)

    print(f"\nMost Frequent Keywords:\n{most_frequent_keywords(df, top_n=50).to_string()}", file=f)

    print(f"\nAnalyze Cooccurrence:\n{analyze_cooccurrence(df)}", file=f)

    print(f"\nAnalyze Frequency:\n{analyze_freq(df)}", file=f)

    wordtree(most_cited_papers(df, 30), 'AB', termo_central='system')

if __name__ == "__main__":
    df = pd.read_csv('all_data_prepared.csv')

    f = open("output.txt", "a", encoding="utf-8")

    run_bibliometric_analysis(df)
