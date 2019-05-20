import glob
import errno
import re

path = './*.txt'
files = glob.glob(path)

nested = {}
i = 0
for name in files:
    try:
        with open(name,'r') as f:
            print(name)
            print(re.findall('\d+',name))
            content = f.readlines()
            year_line = content[1]
            text = content[2]
            nested[name.replace('.\\','')] = []
            
            year_line_splitted = year_line.split(",")
            actual_year = re.findall('\d+',year_line_splitted[1])
            dict_temp = {name:year_line,"date":actual_year}
            nested[name.replace('.\\','')].append(dict_temp)
            
#            print(actual_year)
#            print(content)
            pass # do what you want
    except IOError as exc:
        print("ex")
        if exc.errno != errno.EISDIR:
            raise
            
print(nested)
array = [1,2,3,4]
print(array[2:])