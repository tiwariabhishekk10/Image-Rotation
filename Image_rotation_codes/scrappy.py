from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import requests 
import time
import random

#####

html=urlopen('https://www.shutterstock.com/category/nature')
print(html)

bs=BeautifulSoup(html,'html.parser')

links=[]

images=bs.find_all('img',{'src':re.compile('.jpg')})

for img in images:
    links.append(img['src'])

#####

html2=urlopen('https://www.freeimages.com/search/america-city/2')

bs2=BeautifulSoup(html2,'html.parser')

links2=[]

images2=bs2.find_all('img',{'src':re.compile('.jpg')})

for img in images2:
    links2.append(img['src'])

#####
html3=urlopen('https://www.freeimages.com/search/america-city')

bs3=BeautifulSoup(html3,'html.parser')

links3=[]

images3=bs3.find_all('img',{'src':re.compile('.jpg')})

for img in images3:
    links3.append(img['src'])

#####

html4=urlopen('https://www.shutterstock.com/search/nature?page=2')

bs4=BeautifulSoup(html4,'html.parser')

links4=[]

images4=bs4.find_all('img',{'src':re.compile('.jpg')})


for img in images4:
    links4.append(img['src'])

#####

html5=urlopen('https://www.shutterstock.com/category/animals-wildlife')

bs5=BeautifulSoup(html5,'html.parser')

links5=[]

images5=bs5.find_all('img',{'src':re.compile('.jpg')})


for img in images5:
    links5.append(img['src'])

len(links5)

#####

html6=urlopen('https://www.shutterstock.com/category/religion/spiritualism')

bs6=BeautifulSoup(html6,'html.parser')

links6=[]

images6=bs6.find_all('img',{'src':re.compile('.jpg')})


for img in images6:
    links6.append(img['src'])

len(links6)

#####

html7=urlopen('https://www.shutterstock.com/category/industrial/industrial-buildings')

bs7=BeautifulSoup(html7,'html.parser')

links7=[]

images7=bs7.find_all('img',{'src':re.compile('.jpg')})


for img in images7:
    links7.append(img['src'])

len(links7)

#####

html8=urlopen('https://www.shutterstock.com/category/buildings-landmarks/landmarks-monuments')

bs8=BeautifulSoup(html8,'html.parser')

links8=[]

images8=bs8.find_all('img',{'src':re.compile('.jpg')})


for img in images8:
    links8.append(img['src'])

len(links8)

#####

html9=urlopen('https://www.shutterstock.com/category/sports-recreation/team-sports')

bs9=BeautifulSoup(html9,'html.parser')

links9=[]

images9=bs9.find_all('img',{'src':re.compile('.jpg')})


for img in images9:
    links9.append(img['src'])

len(links9)

#####

links_image=links+links2+links3+links4+links5+links6+links7+links8+links9
len(links_image)

##Image Downloading

#r = requests.get(links_image[0])
#open('1.jpg', 'wb').write(r.content)

for i in range(len(links_image)):
    r = requests.get(links_image[i])
    open(file = 'Image'+str(random.randint(0,1100))+'.jpg', mode= 'wb').write(r.content)
    time.sleep(random.randint(1,5))