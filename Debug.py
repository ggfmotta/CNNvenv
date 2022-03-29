import matplotlib.pyplot as plt


# Plot gray frame
fig, ax = plt.subplots()
plt.clf
im = ax.imshow(imgArray,cmap = 'gray')
plt.show()
plt.savefig(pathOut+'grayframe0.png')
