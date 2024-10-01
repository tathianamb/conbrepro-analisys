import pandas as pd
import re
import pycountry
pd.set_option('display.max_colwidth', None)

university_regex = r"(University|Universidade|Universität|Universidad|Universiti|Univerisity|Universitat|Université|Univ|TU|IET|SNU)"
other_int_regex = r"(NIT|Institute|Inst|Unité|Politecnico|Polytechnic|KU Leuven|College|Coll|Laboratory|Center|Centro|Faculty|École|Consulting|INSEAD|PLC|Lab)"


def extract_institution_country(addresses):
    if pd.isna(addresses):
        return '', ''
    institutions = []
    countries = []

    addr_list = addresses.split(';')
    for addr in addr_list:
        parts = [part.strip() for part in addr.split(',')]
        parts = [part.replace("United States", "USA") for part in parts]

        all_countries = [country.name for country in pycountry.countries]
        all_countries.append("USA")
        all_countries.append("U Arab Emirates")
        all_countries.append("South Korea")
        all_countries.append("Taiwan")
        all_countries.append("Iran")
        re_all_countries = r'\b(' + '|'.join(re.escape(pais) for pais in all_countries) + r')\b'
        countries.extend(re.findall(re_all_countries, parts[-1]))

        if any(re.search(university_regex, part) for part in parts):
            for part in parts:
                if re.search(university_regex, part):
                    institutions.append(part)

        elif any(re.search(other_int_regex, part) for part in parts):
            for part in parts:
                if re.search(other_int_regex, part):
                    institutions.append(part)

    return ';'.join(institutions), ';'.join(countries)



df = pd.read_csv('all_data.csv')

df[['Institution', 'Country']] = df['AD'].apply(extract_institution_country).apply(pd.Series)

df.to_csv('all_data_prepared.csv', index=False)