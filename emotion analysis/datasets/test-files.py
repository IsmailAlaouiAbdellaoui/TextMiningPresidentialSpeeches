# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:49:04 2019

@author: Smail
"""

train_file = open("train.csv","r",encoding="utf-8")
content_train = train_file.readlines()
content_train = [x.strip() for x in content_train] 
train_file.close()
#print(content)

noemo_train = 0
joy_train = 0
surprise_train = 0
sadness_train = 0
anger_train = 0
disgust_train = 0
fear_train = 0
total_train = 0

for item in content_train:
    total_train += 1
    if "noemo" in str(item):
        noemo_train += 1
    if "joy" in str(item):
        joy_train += 1
    if "surprise" in str(item):
        surprise_train += 1
    if "sadness" in str(item):
        sadness_train += 1
    if "anger" in str(item):
        anger_train += 1
    if "disgust" in str(item):
        disgust_train += 1
    if "fear" in str(item):
        fear_train += 1
        

print("Training file : ")        
print("noemo_train",noemo_train/total_train*100)
print("joy_train",joy_train/total_train*100)
print("surprise_train",surprise_train/total_train*100)
print("sadness_train",sadness_train/total_train*100)
print("anger_train",anger_train/total_train*100)
print("disgust_train",disgust_train/total_train*100)
print("fear_train",fear_train/total_train*100)
print(total_train)





test_file = open("test.csv","r",encoding="utf-8")
content_test = test_file.readlines()
content_test = [x.strip() for x in content_test]
test_file.close()

noemo_test = 0
joy_test = 0
surprise_test = 0
sadness_test = 0
anger_test = 0
disgust_test = 0
fear_test = 0
total_test = 0

for item in content_test:
    total_test += 1
    if "noemo" in str(item):
        noemo_test += 1
    if "joy" in str(item):
        joy_test += 1
    if "surprise" in str(item):
        surprise_test += 1
    if "sadness" in str(item):
        sadness_test += 1
    if "anger" in str(item):
        anger_test += 1
    if "disgust" in str(item):
        disgust_test += 1
    if "fear" in str(item):
        fear_test += 1

print("\n\n")
print("Testing file : ")
print("noemo_test",noemo_test/total_test*100)
print("joy_test",joy_test/total_test*100)
print("surprise_test",surprise_test/total_test*100)
print("sadness_test",sadness_test/total_test*100)
print("anger_test",anger_test/total_test*100)
print("disgust_test",disgust_test/total_test*100)
print("fear_test",fear_test/total_test*100)
print(total_test)


dev_file = open("dev.csv","r",encoding="utf-8")
content_dev = dev_file.readlines()
content_dev = [x.strip() for x in content_dev]
dev_file.close()

noemo_dev = 0
joy_dev = 0
surprise_dev = 0
sadness_dev = 0
anger_dev = 0
disgust_dev = 0
fear_dev = 0
total_dev = 0

for item in content_dev:
    total_dev += 1
    if "noemo" in str(item):
        noemo_dev += 1
    if "joy" in str(item):
        joy_dev += 1
    if "surprise" in str(item):
        surprise_dev += 1
    if "sadness" in str(item):
        sadness_dev += 1
    if "anger" in str(item):
        anger_dev += 1
    if "disgust" in str(item):
        disgust_dev += 1
    if "fear" in str(item):
        fear_dev += 1
        
print("\n\n")
print("Dev file : ")
print("noemo_dev",noemo_dev/total_test*100)
print("joy_dev",joy_dev/total_test*100)
print("surprise_dev",surprise_dev/total_test*100)
print("sadness_dev",sadness_dev/total_test*100)
print("anger_dev",anger_dev/total_test*100)
print("disgust_dev",disgust_dev/total_test*100)
print("fear_dev",fear_dev/total_test*100)
print(total_dev)


        

        









        