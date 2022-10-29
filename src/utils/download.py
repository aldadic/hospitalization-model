import requests
import os

# Because covid19-dashboard.ages.at apparently has a weak DH key, we need to lower the security level
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL:@SECLEVEL=1'


def download_files(urls: dict, output_dir: str):
    for name, url in urls.items():
        print('Downloading', url)
        with open(os.path.join(output_dir, name + ".csv"), 'wb') as f:
            f.write(requests.get(url).content)
