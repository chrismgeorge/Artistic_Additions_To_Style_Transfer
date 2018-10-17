import cv2
import os
import numpy

def makeMiniCollage():
    name_new_pic = input()
    directory_path = "./collage_photos/"

    # contains excess files, so don't use length as check if # of pics
    files = os.listdir(directory_path)
    collage_photos = 60
    rows = 10
    cols = 6
    img = numpy.zeros([512*cols,512*rows,3])
    photo_num = 0
    curCol = 0
    curRow = 0
    a = set()

    for filename in files:
        file_path = directory_path+filename
        if (".DS" in file_path): continue

        pic = cv2.imread(file_path, cv2.IMREAD_COLOR)

        dx = (curCol)*512
        dy = (curRow)*512

        # assert correctness
        if ((dx, dy) in a):
            print('Error')
            return None
        a.add((dx, dy))

        for row in range(0, len(pic)):
            for col in range(0, len(pic[0])):
                img[dx+row, dy+col, 0] = pic[row, col, 0]
                img[dx+row, dy+col, 1] = pic[row, col, 1]
                img[dx+row, dy+col, 2] = pic[row, col, 2]

        photo_num += 1

        if (photo_num % cols) == 0:
            curRow += 1
        curCol = photo_num % cols
    print(photo_num)

    cv2.imwrite(name_new_pic+'.jpg', img)

#makeMiniCollage()


def makeMegaCollage():
    directory_path = "./all_stylized_photos/"

    # contains excess files, so don't use length as check if # of pics
    files = os.listdir(directory_path)
    collage_photos = 60
    rows = 62
    cols = 26
    img = numpy.zeros([512*cols,512*(62),3])

    curRow = -1
    a = set()

    for folder_name in files:
        if (".DS" in folder_name): continue
        curRow += 1
        curCol = 0
        for filename in os.listdir(directory_path+folder_name):
            file_path = directory_path+folder_name+"/"+filename
            if (".DS" in file_path or "seg" in file_path): continue

            pic = cv2.imread(file_path, cv2.IMREAD_COLOR)

            dx = (curCol)*512
            dy = (curRow)*512

            # assert correctness
            if ((dx, dy) in a):
                print('Error')
                return None
            a.add((dx, dy))

            for row in range(0, len(pic)):
                for col in range(0, len(pic[0])):

                    img[dx+row, dy+col, 0] = pic[row, col, 0]
                    img[dx+row, dy+col, 1] = pic[row, col, 1]
                    img[dx+row, dy+col, 2] = pic[row, col, 2]

            curCol += 1
        print("Finished... ", str(curRow)+"/63")

    cv2.imwrite('the_meg.jpg', img)

#makeMegaCollage()

# pic = cv2.imread('the_meg.jpg', cv2.IMREAD_COLOR)
# cv2.imwrite('the_meg_2.jpg', pic)

# pic = cv2.imread('the_meg_2.jpg', cv2.IMREAD_COLOR)
# img = numpy.zeros([len(pic),len(pic[0])-512,3])

# for row in range(0, len(pic)):
#     print("row", str(row), " out of ", str(len(pic)))
#     for col in range(0, len(pic[0])):
#         if (col < len(pic[0])-512):
#             img[row, col, 0] = pic[row, col, 0]
#             img[row, col, 1] = pic[row, col, 1]
#             img[row, col, 2] = pic[row, col, 2]

# cv2.imwrite('the_meg_3.jpg', img)

