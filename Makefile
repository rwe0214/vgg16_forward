SRCDIR = src
OUTDIR ?= .out
CC	= g++
CFLAGS	=-std=c++11 -g -Wall

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(OUTDIR)/%.o, $(SRCS)) 
EXE = vgg16

all: $(EXE)

$(EXE) : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ 

$(OUTDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -c -o $@ $<
	
SHELL_HACK := $(shell mkdir -p $(OUTDIR))

.PHONY: all clean run debug OUT

format:
	clang-format -i $(SRCDIR)/*.cpp $(SRCDIR)/*.h

run:
	./$(EXE)

debug:
	gdb -q $(EXE)

clean:
	rm -f $(OUTDIR)/* ./$(EXE)
	rmdir ./$(OUTDIR)
