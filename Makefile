CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

all:
	$(NVCC) main.cu -o matrix_mul

run:
	./matrix_mul

clean:
	rm -f matrix_mul
