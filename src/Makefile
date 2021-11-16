SRCS := main.c
OBJS := $(SRCS:.c=.o)
CC := gcc
CFLAGS := -mavx2

matmul: main.o
	$(CC) -o $@ $(OBJS)

clean:
	-rm $(OBJS)
	rm matmul
