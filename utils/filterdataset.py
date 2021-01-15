import os, sys

files = os.listdir("SIDOD/" + sys.argv[1])
start = sys.argv[2]


def filter():
    for file in files:
        if file.split('.')[-2] not in ['depth', 'left', 'right'] or file.split('.')[-1] != 'png':
            os.remove("SIDOD/" + sys.argv[1] + "/" + file)

def renumber():
    for file in files:
        split_file = file.split('.')
        split_file[0] = str(int(split_file[0]) + int(start))
        #split_file[0] = split_file[0][2:]
        os.rename("SIDOD/" + sys.argv[1] + "/" + file, "SIDOD/" + sys.argv[1] + "/" + ".".join(split_file))


#filter()
files = os.listdir("SIDOD/" + sys.argv[1])
renumber()
