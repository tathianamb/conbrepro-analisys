import pandas as pd
import os

# Caminho da pasta onde os arquivos estão localizados
folder_path = "DATASET2023"

# Lista de arquivos CSV (com caminho completo)
files = [
    os.path.join(folder_path, "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023.csv"),
    os.path.join(folder_path, "INMET_CO_GO_A002_GOIANIA_01-01-2023_A_31-12-2023.csv"),
    os.path.join(folder_path, "INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_31-12-2023.csv"),
    os.path.join(folder_path, "INMET_NE_PI_A312_TERESINA_01-01-2023_A_31-12-2023.csv")
]


def preprocess_file(file_path):
    # Carrega o arquivo CSV pulando as primeiras 8 linhas e lendo a linha 9 como o header
    df = pd.read_csv(file_path, delimiter=';', encoding='latin1', skiprows=8)

    # Salva o DataFrame sem as primeiras 8 linhas (não processado)
    raw_path = os.path.join(folder_path, os.path.splitext(os.path.basename(file_path))[0] + '_no_description.csv')
    df.to_csv(raw_path, index=False)

    # Renomeia as colunas para consistência (remove espaços e converte para minúsculas)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    columns_to_keep = ['data', 'hora_utc', 'radiacao_global_(kj/m²)']
    df = df[columns_to_keep]
    df.columns = ['data', 'hora', 'radiacao_global']

    # Tratamento de valores
    df = df.replace('////', pd.NA)
    df['radiacao_global'] = df['radiacao_global'].str.replace(',', '.').astype(float)
    df.fillna({'radiacao_global': 0}, inplace=True)
    df['radiacao_global'] = pd.to_numeric(df['radiacao_global'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['data'].astype(str) + ' ' + df['hora'].astype(str).str.replace(' UTC', ''), format='%Y/%m/%d %H%M')
    df = df.drop(columns=['data', 'hora'])
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.set_index('datetime')

    processed_file_path = os.path.join(folder_path, os.path.splitext(os.path.basename(file_path))[0] + '_processed.csv')
    df.to_csv(processed_file_path, index=True)
    print(f"Arquivo processado salvo em: {processed_file_path}")


# Pré-processa cada arquivo
for file in files:
    preprocess_file(file)
