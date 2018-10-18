import os

# directory_path = "./mom/"
# i = 0
# for filename in os.listdir(directory_path):
#     os.rename(directory_path+filename, directory_path+str(i)+".jpg")
#     i += 1


directory_path = "./collage_photos/"
text = ""
directory = os.listdir(directory_path)

for filename in directory:
    if (".png" in filename):
        text += '<p align="center">  <img width="100" height="100" src="'+filename+'?"> </p>'
        text += '<p align="center">'+filename+'</p>'
        text += "+\n---\n"

print(text)
