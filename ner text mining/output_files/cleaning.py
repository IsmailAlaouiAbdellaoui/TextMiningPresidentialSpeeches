# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:57:56 2019

@author: Smail
"""

raw_file = open('output_ner_3_last_presidents.txt','r')
content = raw_file.readlines()
content = [x.strip() for x in content] 
raw_file.close()
print(content[27])
print("\n")
first_part = content[27].split(" ")[:2]
location_part = content[27][content[27].find("{")+1:content[27].find("}")]
location_part = location_part.replace(".'","'")
location_part = location_part.replace(",'","'")
location_part = location_part.replace(":'","'")
location_part = location_part.replace("?'","'")
location_part = location_part.replace(";'","'")
location_part = location_part.replace("United States of America","United States")
location_part = location_part.replace("'UNITED STATES","United States")
temp = []
location_part = location_part.replace("'","")
location_part = location_part.replace('"','')
print(location_part)
print(type(location_part))
splitted = location_part.split(",")

print("\n")
print(type(splitted))
print(set(splitted))


#
output_file = open('cleaned_output_3_last_presidents.txt','w')

for item in content:
    first_part = item.split(" ")[:2]
    location_part = item[item.find("{")+1:item.find("}")]
    location_part = location_part.replace(".'","'")
    location_part = location_part.replace(",'","'")
    location_part = location_part.replace(":'","'")
    location_part = location_part.replace("?'","'")
    location_part = location_part.replace(";'","'")
    location_part = location_part.replace("United States of America","United States")
    location_part = location_part.replace("'UNITED STATES","United States")
    location_part = location_part.replace("'","")
    location_part = location_part.replace('"','')
    splitted = location_part.split(",")
    output_file.write(str(first_part) + " "+ str(set(splitted)))
    output_file.write("\n")
    
output_file.close()

    