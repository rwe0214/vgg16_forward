# VGG16

[![hackmd-github-sync-badge](https://hackmd.io/bC3GmR5uSHGdvz9DxKV-xw/badge)](https://hackmd.io/bC3GmR5uSHGdvz9DxKV-xw)




Implement a simple vgg16 network architecture only with forward pass



## How to run



```shell
$ make
g++ -std=c++11 -g -Wall -c -o .out/main.o src/main.cpp
g++ -std=c++11 -g -Wall -c -o .out/network.o src/network.cpp
g++ -std=c++11 -g -Wall -c -o .out/utils.o src/utils.cpp
g++ -std=c++11 -g -Wall -c -o .out/vgg16.o src/vgg16.cpp
g++ -std=c++11 -g -Wall -o vgg16 .out/main.o .out/network.o .out/utils.o .out/vgg16.o
$ make run
```



## Result



```shell
$ make run
./vgg16
Layer		Memory size			Param #		MAC #
===========================================================================
conv1_1		1*224*224*3 = 150528    	1792    	86704128
conv1_2		1*224*224*64 = 3211264   	36928   	1849688064
max_pooling_1	1*224*224*64 = 3211264   	0
conv2_1		1*112*112*64 = 802816    	73856   	924844032
conv2_2		1*112*112*128 = 1605632   	147584  	1849688064
max_pooling_2	1*112*112*128 = 1605632   	0
conv3_1		1*56*56*128 = 401408    	295168  	924844032
conv3_2		1*56*56*256 = 802816    	590080  	1849688064
conv3_3		1*56*56*256 = 802816    	590080  	1849688064
max_pooling_3	1*56*56*256 = 802816    	0
conv4_1		1*28*28*256 = 200704    	1180160 	924844032
conv4_2		1*28*28*512 = 401408    	2359808 	1849688064
conv4_3		1*28*28*512 = 401408    	2359808 	1849688064
max_pooling_4	1*28*28*512 = 401408    	0
conv5_1		1*14*14*512 = 100352    	2359808 	462422016
conv5_2		1*14*14*512 = 100352    	2359808 	462422016
conv5_3		1*14*14*512 = 100352    	2359808 	462422016
max_pooling_5	1*14*14*512 = 100352    	0
fc1_4096	1*1*1*25088 = 25088       	102764544	16777216
fc2_4096	1*1*1*4096 = 4096        	16781312	16777216
fc3_1000	1*1*1*4096 = 4096        	4097000 	1000000
===========================================================================
```