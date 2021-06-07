all:
	nvcc main.cu func.cu -o cluster
clean:
	rm -rf cluster
