objects = test.o inference.o

all: test

@PHONY clean:
	rm -f *.o *.ii *.cubin *.fatbin.c *.fatbin *.cudafe1.* *.module_id *.ptx *.reg.c test

test: test.cu inference.cu inference.h
	nvcc \
		-O3 -lineinfo -Xcompiler -rdynamic --use_fast_math -keep -std=c++17 --ptxas-options=-v \
		-gencode arch=compute_86,code=sm_86  \
		-gencode arch=compute_89,code=sm_89  \
		-gencode arch=compute_90,code=sm_90  \
		-o $@ test.cu inference.cu