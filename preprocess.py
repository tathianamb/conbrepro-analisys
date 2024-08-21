import pandas as pd
import numpy as np

"""
    Na programação utilizamos como guia o chamado "Clean Code", que é o que rege a ideia que você me pediu.

    Portanto, o primeiro tópico que deixo é:
        1- Não utilize 'comentários desnecessários', o que não é DOCUMENTAÇÃO;
        2- Desenvolva tudo em INGLÊS - em PT-BR somente a DOCUMENTAÇÃO;
        3- A função tem que exercer somente uma etapa. Ex: "createHandleAndGetResult() deveriam ser createData() / handleData() / getResultOfData()
            3.1- Se os parâmetros das funções se repetem, então deveria ser uma variável da classe. Ex: As funções 'differentiate' e 'reverse_differentiation'
                    recebem 'df' e 'name', mas isso deveria fazer parte das variáveis da classe. Pois a ideia de classe é ser ESPECÍFICA e não GENÉRICA.
            3.2- Na 'main', cada um dos blocos deveria se tornar uma função PRIVADA.
        4- Dê valores padrões para os parâmetros que dificilmente mudem (ex: 'column_name' no construtor da classe Differentiator)


    LUFI UUUUUUUU    
"""


class Differentiator:
    def __init__(self, column_name, shift=1):
        """
        Inicializa a classe Differentiator.

        Args:
            column_name (str): Nome da coluna para diferenciação.
            shift (int): Número de períodos para o deslocamento.
        """
        self.column_name = column_name
        self.shift = shift
        self.initial_values = {}

    def differentiate(self, df, name):
        """
        Aplica a diferenciação à coluna especificada do DataFrame.

        Args:
            df (pd.DataFrame): DataFrame contendo os dados a serem diferenciados.
            name (str): Nome do conjunto de dados (treinamento, validação, teste).

        Returns:
            pd.DataFrame: DataFrame com a coluna diferenciada.
        """
        df_diff = df.copy()
        if name not in self.initial_values:
            self.initial_values[name] = df_diff[self.column_name].head(self.shift).tolist()

        df_diff[self.column_name] = df_diff[self.column_name] - df_diff[self.column_name].shift(self.shift)
        df_diff = df_diff.iloc[self.shift:]
        return df_diff

    def reverse_differentiation(self, df_diff, name):
        """
        Reverte a diferenciação aplicada anteriormente à coluna especificada do DataFrame.

        Args:
            df_diff (pd.DataFrame): DataFrame contendo os dados diferenciados.
            name (str): Nome do conjunto de dados (treinamento, validação, teste).

        Returns:
            pd.DataFrame: DataFrame com a coluna revertida.
        """
        df = df_diff.copy()
        if name not in self.initial_values:
            raise ValueError(f"Differenciação não aplicada para o conjunto {name}.")

        original_length = len(df) + self.shift
        reverted_values = np.cumsum(self.initial_values[name] + df[self.column_name].tolist()).tolist()

        if len(reverted_values) != original_length:
            raise ValueError(
                f"O comprimento revertido não corresponde ao comprimento original esperado: {original_length}.")

        freq = df.index.freq
        start_period = df.index[0] - self.shift
        end_period = df.index[-1]
        reverted_df_index = pd.period_range(start=start_period, end=end_period, freq=freq)
        reverted_df = pd.DataFrame(data=reverted_values, index=reverted_df_index, columns=df_diff.columns)
        return reverted_df


# Exemplo de uso
if __name__ == "__main__":
    '''df2 = pd.DataFrame({'A': [1, 2, 3]}, index=pd.period_range('2023-01', '2023-03', freq='M'))
    differentiator = Differentiator(column_name='A', shift=1)
    df2_diff = differentiator.differentiate(df2, 'df2')
    print("\nDados df2_diff Diferenciados:")
    print(df2_diff)
    df2_reverted = differentiator.reverse_differentiation(df2_diff, 'df2')
    print("\nDados df2_reverted Revertidos:")
    print(df2_reverted)'''
