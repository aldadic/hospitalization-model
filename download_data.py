from src.utils import download_files

urls = {
    "hospitalization": "https://covid19-dashboard.ages.at/data/Hospitalisierung.csv",
    "age_groups": "https://covid19-dashboard.ages.at/data/CovidFaelle_Altersgruppe.csv"
}

if __name__ == '__main__':
    download_files(urls, output_dir='./data')
