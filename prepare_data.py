import pandas as pd


def ris_to_dataframe(ris_file):
    # Lê o arquivo RIS
    with open(ris_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Divide o conteúdo em registros
    records = content.strip().split('ER  -')  # 'ER  -' indica o fim de um registro

    # Cria uma lista para armazenar os dicionários
    data = []

    for record in records:
        if record.strip():  # Ignora registros vazios
            entry = {}
            # Divide o registro em linhas
            lines = record.strip().split('\n')
            for line in lines:
                tag = line[:2].strip()
                value = line[5:].strip()

                if tag in entry:
                    entry[tag] += f"; {value}"  # Concatena com ';'
                else:
                    entry[tag] = value

            # Adiciona o registro ao conjunto de dados
            data.append(entry)

    # Converte a lista de dicionários em um DataFrame
    df = pd.DataFrame(data)
    return df


ris_file_paths = ['scopus.ris', 'savedrecs.ris']  # Substitua pelos seus arquivos

# DataFrame vazio para armazenar todos os dados
all_data = pd.DataFrame()

for ris_file_path in ris_file_paths:
    df = ris_to_dataframe(ris_file_path)
    df.to_csv(f'{ris_file_path}.csv', index=False)
    all_data = pd.concat([all_data, df], ignore_index=True)

all_data.to_csv('alldata.csv', index=False)