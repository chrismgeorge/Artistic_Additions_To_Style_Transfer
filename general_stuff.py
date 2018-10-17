import os

directory_path = "./mom/"
i = 0
for filename in os.listdir(directory_path):
    os.rename(directory_path+filename, directory_path+str(i)+".jpg")
    i += 1
