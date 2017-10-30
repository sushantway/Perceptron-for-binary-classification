#Name: Sushant Chaudhari

import random
import os
import struct
import numpy as np

class Perceptron():
    def __init__(self,learn_speed = 0, num_weights = 0):
        self.speed = learn_speed
        self.weights = []
        for x in range(0, num_weights):
    		self.weights.append(random.random()*2-1)
        #print(self.weights)

    def feed_forward(self, inputs):
  		sum = 0
  		for x in range(0, len(self.weights)):
  			sum += self.weights[x] * inputs[x]
  		#print(sum)
  		return self.activate(sum)
    
    def activate(self,num):
   		#print(num)
  		if num > 0:
  			return 1
  		return -1


    
    def train(self, inputs, desired_output):
    	if(desired_output == 6):
    		do = -1
    	else:
    		do = 1
      	guess = self.feed_forward(inputs)
      	error = do - guess
      
      	for x in range(0, len(self.weights)):
        	self.weights[x] += error*inputs[x]*self.speed

    def printWeight(self):
    	print(self.weights) 
    	print(self.speed)

    def plotWeight(self):
    	b = np.reshape(self.weights, (28, 28))
    	print(b)
    	self.show(b)

    def show(self,image):
    	from matplotlib import pyplot
    	import matplotlib as mpl
    	fig = pyplot.figure()
    	ax = fig.add_subplot(1,1,1)
    	imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    	imgplot.set_interpolation('nearest')
    	ax.xaxis.set_ticks_position('top')
    	ax.yaxis.set_ticks_position('left')
    	pyplot.show()

    def plotWeightPositive(self):
    	x = 0
    	for x in range(len(self.weights)):
    		if(self.weights[x]<0):
    			self.weights[x] = 0
    	b = np.reshape(self.weights, (28, 28))
    	print(b)
    	self.show(b)

    def plotWeightNegative(self):
    	x = 0
    	for x in range(len(self.weights)):
    		if(self.weights[x]>0):
    			self.weights[x] = 0
    	b = np.reshape(self.weights, (28, 28))
    	print(b)
    	self.show(b)


#function to read data from file
def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def c_question_graph_plot():
	#The below values are collected by running the code for different number of iterations
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    y = [98.0,98.3,98.8,98.9,98.8,98.8,98.9,99.0,99.0,99.4]
    x = [1,2,3,4,5,6,7,8,9,10]
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.plot(x, y, linewidth=2.0)
    plt.show()
    print("plotted")

def f_question_graph_plot():
	#The below values are collected by running the code for different number of iterations
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    y = [96.4,97.9,96.8,96.5,93.9,93.0,97.3,98.0,97.9,97.3]
    x = [1,2,3,4,5,6,7,8,9,10]
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.plot(x, y, linewidth=2.0)
    plt.show()
    print("plotted")

def h_question_graph_plot():
	#The below values are collected by running the code for different number of iterations
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    y = [91.0,91.1,92.4,93.7,94.0,93.6,95.4,94.1,95.3,94.8]
    x = [1,2,3,4,5,6,7,8,9,10]
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.plot(x, y, linewidth=2.0)
    plt.show()
    print("plotted")

def g_question_graph_plot():
	#The below values are collected by running the code for different number of iterations
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    y = [52.0,95.0,98.6,99.1,98.7,98.8,98.5,99.3,99.0,99.4]
    x = [1,2,3,4,5,6,7,8,9,10]
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.plot(x, y, linewidth=2.0)
    plt.show()
    print("plotted")

def i_question_graph_plot():
	#The below values are collected by running the code for different number of iterations
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    y = [93.38,93.16,94.88,93.88,93.94,93.11,97.77,98.0,97.83,97.88]
    x = [1,2,3,4,5,6,7,8,9,10]
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.plot(x, y, linewidth=2.0)
    plt.show()
    print("plotted")

def main():
    mnist_train = read("training")
    mnist_test = read("testing")
    binaryInput1 = 1
    binaryInput2 = 6
    NumOfIterations = 10
    f_question = False
    g_question = False

    train_labels = []
    train_images = []
    test_labels = []
    test_images = []

    cnt1 = 0
    cnt6 = 0

    #generate training dataset
    if(g_question == False):
    	for i in range(60000):
    		if(cnt1 == 500 and cnt6 == 500):
    			break
        	label_image = mnist_train.next()
        	if(label_image[0] == binaryInput1 or label_image[0] == binaryInput2):
        		if(label_image[0] == binaryInput1 and cnt1!=500):
        			cnt1 += 1
        			train_labels.append(label_image[0])
        			train_images.append(label_image[1])
        		if(label_image[0] == binaryInput2 and cnt6!=500):
        			cnt6 += 1
        			train_labels.append(label_image[0])
        			train_images.append(label_image[1])

    if(g_question):
    	for i in range(60000):
    		if(cnt1 == 500):
    			break
        	label_image = mnist_train.next()
        	if(label_image[0] == binaryInput1):
        		if(label_image[0] == binaryInput1 and cnt1!=500):
        			cnt1 += 1
        			train_labels.append(label_image[0])
        			train_images.append(label_image[1])
       	i = 0
        for i in range(60000):
    		if(cnt6 == 500):
    			break
        	label_image = mnist_train.next()
        	if(label_image[0] == binaryInput2):
        		if(label_image[0] == binaryInput2 and cnt6!=500):
        			cnt6 += 1
        			train_labels.append(label_image[0])
        			train_images.append(label_image[1])


    #generate testing dataset
    cnt1 = 0
    cnt6 = 0
    i = 0
    for i in range(60000):
       	if(cnt1 == 500 and cnt6 == 500):
       		#print("Final Counts")
       		#print(cnt1)
       		#print(cnt6)
       		break
        label_image1 = mnist_test.next()
        if(label_image1[0] == binaryInput1 or label_image1[0] == binaryInput2):
        	if(label_image1[0] == binaryInput1 and cnt1!=500):
        		cnt1 += 1
        		#print(cnt1)
        		test_labels.append(label_image1[0])
        		test_images.append(label_image1[1])
        	if(label_image1[0] == binaryInput2 and cnt6!=500):
        		cnt6 += 1
        		#print(cnt6)
        		test_labels.append(label_image1[0])
        		test_images.append(label_image1[1])

    print("Train Labels:")
    print(train_labels)
    
    rand = 0
    #Train the perceptron
    if(f_question):
    	for t in range(len(train_labels)):
    		if(bool(random.getrandbits(1))):
    			rand += 1
    			if(rand == 100):
    				break
    			if(train_labels[t] == 1):
    				train_labels[t] = 6
    			else:
    				train_labels[t] = 1
   		print("Rand:",rand)
    	print("Train Labels after flipping:")
    	print(train_labels)
    
    p = Perceptron(0.01,784)
    for _ in range(NumOfIterations):
    	for x in range(len(train_images)):
    		a = np.array(train_images[x])
    		y = np.ravel(a)
    		p.train(y,train_labels[x])

    

    p.printWeight()
    #print("Test Labels:")
    #print(test_labels)
    x = 0
    count1 = 0.00
    count6 = 0.00
    for x in range(len(test_images)):
    		a = np.array(test_images[x])
    		y = np.ravel(a)
    		val = p.feed_forward(y)
    		#print(val)
    		#print(test_labels[x])
    		if(val == 1 and test_labels[x] == binaryInput1):
    			count1 += 1
    		if(val == 1 and test_labels[x] == binaryInput2):
    			print("Wrongly classified 1 as 6")
    			#print(test_images[x])
    		if(val == -1 and test_labels[x] == binaryInput2):
    			count6 += 1
    		if(val == -1 and test_labels[x] == binaryInput1):
    			print("Wrongly classified 6 as 1")
    			#print(test_images[x])
    #print(count1)
    #print(count6)
    accuracy = ((count1 + count6)/1000.00)*100.00
    print("Accuracy is:",accuracy)

    #c_question_graph_plot()
    #p.plotWeight()
    #f_question_graph_plot()
    #h_question_graph_plot()
    #g_question_graph_plot()
    #i_question_graph_plot()
    #p.plotWeightPositive()
    #p.plotWeightNegative()
         	

if __name__== "__main__":
  main()