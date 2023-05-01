import sys

dataset = sys.argv[1]
path_export = sys.argv[2]

for i in range(1000):
    name_export = "{}explore_iteration_{}.csv".format(path_export, i)

    open_doc = open("config_{}.txt".format(i), 'w')
    open_doc.write(dataset+"\n")
    open_doc.write(name_export+"\n")
    open_doc.write(str(i))
    open_doc.close()


