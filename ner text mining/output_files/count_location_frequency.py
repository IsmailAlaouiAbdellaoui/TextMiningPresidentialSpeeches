# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:17:14 2019

@author: Smail
"""

import nltk
import matplotlib.pyplot as plt



raw_file = open('cleaned_output_3_last_presidents.txt','r')
content = raw_file.readlines()
content = [x.strip() for x in content] 
raw_file.close()
#print(content)
#print(content[0])

#location_part = content[0][content[0].find("{")+1:content[0].find("}")]
#location_part = location_part.replace("' ","'")
#print(location_part)
#print(type(location_part))
#splitted = location_part.split(",")
#for item in splitted:
#    all_locations.append(item.replace("'","").strip())
#    
#print(all_locations)

all_locations = []

raw_frequency_file = open("frequency_count_3_last_presidents.txt","w")


for item in content:
    location_part = item[item.find("{")+1:item.find("}")]
    location_part = location_part.replace("' ","'")
    splitted = location_part.split(",")
    for item in splitted:
        if ".txt" in item:
            continue 
        else:
            all_locations.append(item.replace("'","").strip())
    
fdist = nltk.FreqDist(all_locations)
list_locations_labels =[]
list_frequency = [] 

for word, frequency in fdist.most_common():
#    print(u'{} : {}'.format(word, frequency))
    raw_frequency_file.write(u'{} : {}'.format(word, frequency))
    raw_frequency_file.write("\n")
    list_locations_labels.append(word)
    list_frequency.append(frequency)

raw_frequency_file.close()

total = sum(list_frequency)




all_locations_without_us = []
for item in content:
    location_part = item[item.find("{")+1:item.find("}")]
    location_part = location_part.replace("' ","'")
    splitted = location_part.split(",")
    for item in splitted:
        if ".txt" in item:
            continue 
        if "United States" in item:
            continue
        else:
            all_locations_without_us.append(item.replace("'","").strip())
            
fdist_without_us = nltk.FreqDist(all_locations_without_us)
list_locations_labels_without_us =[]
list_frequency_without_us = [] 

for word, frequency in fdist_without_us.most_common(20):
    print(u'{} : {}'.format(word, frequency))
    list_locations_labels_without_us.append(word)
    list_frequency_without_us.append(frequency)

total_without_us = sum(list_frequency_without_us)



#fig1, ax1 = plt.subplots()
#ax1.pie(list_frequency,  labels=list_locations_labels, autopct=lambda p: '{:.0f}'.format(p * total / 100),
#        shadow=True, startangle=90)

#ax1.pie(list_frequency_without_us,  labels=list_locations_labels_without_us, autopct=lambda p: '{:.0f}'.format(p * total_without_us / 100),
#        shadow=True, startangle=90)

#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
#plt.show()
    