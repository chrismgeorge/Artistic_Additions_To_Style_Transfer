import os

# directory_path = "./mom/"
# i = 0
# for filename in os.listdir(directory_path):
#     os.rename(directory_path+filename, directory_path+str(i)+".jpg")
#     i += 1


directory_path = "./all_stylized_photos/"

directory = os.listdir(directory_path)

# for folder_name in directory:
#     if (".DS" in folder_name): continue
#     text = ""
#     folder_dir = directory_path+folder_name+"/"
#     pics = os.listdir(folder_dir)
#     for filename in pics:
#         if (".png" in filename):
#             text += '<p align="center">  <img width="100" height="100" src="'+filename+'?"> </p>'
#             text += '<p align="center">'+filename+'</p>'
#             text += "\n\n***\n\n"
#     new_read_me = folder_dir + "README.md"
#     file = open(new_read_me, "w")
#     file.write(text)

print(text)
