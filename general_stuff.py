import os

# directory_path = "./mom/"
# i = 0
# for filename in os.listdir(directory_path):
#     os.rename(directory_path+filename, directory_path+str(i)+".jpg")
#     i += 1


directory_path = "./regular_photos/"
text = ""
for filename in os.listdir(directory_path):
    if (".jpg" in filename):
        text += '<p align="center">  <img width="100" height="100" src="'+filename+'?"> </p>'
        text += '<p align="center">'+filename+'</p>'
        text += "\n"

print(text)
