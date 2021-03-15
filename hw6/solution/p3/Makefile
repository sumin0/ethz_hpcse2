CUFLAGS=-O3 -std=c++14 --compiler-options "-Wall -Wextra -fopenmp"

.PHONY: all clean

all: overlap overlap_profile
	@true

clean:
	rm -f overlap overlap_profile

%: %.cu utils.h
	nvcc $(CUFLAGS) $< -o $@
