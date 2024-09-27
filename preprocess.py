import pandas as pd


instituicao_regex = r"(University|Universidade|Universität|Universidad|Unité|École|Ecole|Faculty|Department|Corporation|CanmetENERGY|College|Politecnico|Université|Center|Universitat|Institute|Ministry)"

def extract_institution_country(addresses):
    if pd.isna(addresses):
        return '', ''
    institutions = []
    countries = []

    addr_list = addresses.split(';')
    for addr in addr_list:
        parts = [part.strip() for part in addr.split(',')]
        if len(parts) >= 4:
            institution = parts[-4]
            country = parts[-1]
            institutions.append(institution)
            countries.append(country)
        else:
            institutions.append(None)
            countries.append(None)

    return '; '.join(filter(None, institutions)), '; '.join(filter(None, countries))



df = pd.read_csv('alldata.csv')

df[['Institution', 'Country']] = df['AD'].apply(extract_institution_country).apply(pd.Series)

df.to_csv('alldata_prepared.csv', index=False)