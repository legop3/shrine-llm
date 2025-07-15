from bs4 import BeautifulSoup

import requests
import urllib.request

request = requests.get('https://files.in5net.io/discord/messages')

data = request.text
soup = BeautifulSoup(data)

for link in soup.find_all('a'):
    print(link.get('href'))
    try:
        urllib.request.urlretrieve(f"https://files.in5net.io{link.get('href')}", f"realdata/{link.get('href').split("/")[3]}")
    except:
        print('url not downloadable')