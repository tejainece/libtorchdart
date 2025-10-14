build:
	#g++ -w -I ../libtorch_linux/include/torch/csrc/api/include/ -I ../libtorch_linux/include/ -L ../libtorch_linux/lib/ -ltorch_cuda -lc10 -ltorch -lc10_cuda -o main ctorch.cpp
	gcc -I ../src/include -L ../src/build/ main.c -ltorchffi -o main.exe