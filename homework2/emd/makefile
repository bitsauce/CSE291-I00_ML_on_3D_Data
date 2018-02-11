nvcc = /usr/local/cuda-9.1/bin/nvcc
cudalib = /usr/local/cuda-9.1/lib64/
tensorflow = /home/marcus/anaconda3/lib/python3.6/site-packages/tensorflow/include

all: tf_emddistance_so.so 
.PHONY : all

tf_emddistance_so.so:  tf_emddistance.cpp
	g++ -std=c++11 tf_emddistance.cpp  -o tf_emddistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


