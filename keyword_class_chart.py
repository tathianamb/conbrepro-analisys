# Passo 1: Importar as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Passo 2: Ler o arquivo CSV
df = pd.read_csv('keywords_class.csv', sep=';')

# Passo 3: Criar um dicionário de frequências das palavras
frequencies = dict(zip(df['Palavra-chave'], df['Valor']))

# Passo 4: Mapear cada palavra à sua classe
palavra_classe = dict(zip(df['Palavra-chave'], df['Classe']))

# Passo 5: Definir cores específicas para cada classe
# Obter lista de classes únicas
classes_unicas = df['Classe'].unique()

# Definir cores para cada classe (personalize as cores conforme desejar)
cores_classe = {
    classe: cor for classe, cor in zip(classes_unicas, ['#6d238d', '#235a8d', '#A5C724'])
}

# Passo 6: Criar uma função de coloração
def get_color(word, **kwargs):
    classe = palavra_classe.get(word)
    return cores_classe.get(classe, 'black')  # Retorna 'black' se a classe não for encontrada

# Passo 7: Gerar a nuvem de palavras
wordcloud = WordCloud(width=1600, height=800, background_color='white', color_func=get_color)

# Gerar a nuvem de palavras a partir das frequências
wordcloud.generate_from_frequencies(frequencies)

# Passo 8: Exibir a nuvem de palavras
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
wordcloud.to_file('wordcloud_keywords.png')
