#SHELL := /bin/bash
CC=gcc-7
NVCC=nvcc
CFLAGS=  -O3  -std=c99 -lm
CUDAFLAGS= --compiler-options -Wall
EXE=$(SDIR)/v0 $(SDIR)/v1 $(SDIR)/v2 $(SDIR)/v3
LIBS=$(LDIR)/v0.a $(LDIR)/v1.a $(LDIR)/v2.a $(LDIR)/v3.a
SDIR=./src
LDIR=./lib
IDIR=./inc
MAIN=main.cu
INCLUDES= -I$(IDIR)
LIBRARIES= -L$(LDIR)


all:	$(EXE)

lib:	$(LIBS)

$(SDIR)/v0: $(SDIR)/v0.c
	$(CC) $(CFLAGS) -o $@ $^

$(SDIR)/v1: $(LDIR)/v1.a $(SDIR)/$(MAIN)
	$(NVCC) $(INCLUDES) -o $@ $^ $(CUDAFLAGS)

$(SDIR)/v2: $(LDIR)/v2.a $(SDIR)/$(MAIN)
	$(NVCC) $(INCLUDES) -o $@ $^ $(CUDAFLAGS)

$(SDIR)/v3: $(LDIR)/v3.a $(SDIR)/$(MAIN)
	$(NVCC) $(INCLUDES) -o $@ $^ $(CUDAFLAGS)


$(LDIR)/%.a: $(SDIR)/%.o
	ar rcs $@ $^


$(SDIR)/v1.o: $(SDIR)/v1.cu
	$(NVCC) -o $@ -c $^ $(CUDAFLAGS)

$(SDIR)/v2.o: $(SDIR)/v2.cu
	$(NVCC) -o $@ -c $^ $(CUDAFLAGS)

$(SDIR)/v3.o: $(SDIR)/v3.cu
	$(NVCC) -o $@ -c $^ $(CUDAFLAGS)




clean:
	rm -f $(SDIR)/*.o $(EXE) $(LIBS)
