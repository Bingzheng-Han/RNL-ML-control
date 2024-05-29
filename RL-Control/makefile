all : build channel_2a

build : builder.py 
	python builder.py

channel_2a : channel_2a.f90 module.f90
	ifort -c module.f90
	ifort -o channel_2a -L./ plugin.so channel_2a.f90 module.f90 -r8 -O3 -qopenmp -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lmkl_intel_thread