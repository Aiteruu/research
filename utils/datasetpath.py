import pandas as pd  

v = []

for i in range(12000):
    v.append(["{}.right.png".format(i), "{}.right.depth.png".format(i)])   
    v.append(["{}.left.png".format(i), "{}.right.left.png".format(i)]) 

df = pd.DataFrame(v)
df.to_csv('filepaths.csv', index=False)