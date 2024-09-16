import bibtexparser

def read_bib_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as bibfile:
        bib_database = bibtexparser.load(bibfile)
        return bib_database

def display_entries(bib_database):
    for entry in bib_database.entries:
        print(f"Entry Type: {entry.get('ENTRYTYPE', 'N/A')}")
        for key, value in entry.items():
            if key != 'ENTRYTYPE':
                print(f"  {key}: {value}")
        print()

if __name__ == "__main__":
    file_path = 'exemplo.bib'  # Substitua pelo caminho para o seu arquivo .bib
    bib_database = read_bib_file(file_path)
    display_entries(bib_database)
