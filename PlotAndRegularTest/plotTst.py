import numpy as np
import matplotlib.pyplot as plt
import os
import re

loss = np.zeros(10000)

log_file = open("log.txt","r")
count=0
for index ,line in enumerate(log_file):
    count+=1
    m = re.match(r'(.*?)(loss = )([\d\.]*)', line)
    loss_str = m.group(3)
    loss[index] = float(loss_str)


epcho = np.linspace(1,count,count)
print(epcho)
plt.plot(epcho,loss[0:count])
plt.ylabel("loss")
plt.xlabel("training step")
plt.savefig('a.jpg')
plt.show()





log_file.close()