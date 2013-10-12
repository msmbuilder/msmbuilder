all: _vmhmm.so

_vmhmm.o: _vmhmm.c
	gcc -fPIC -g -c -Wall -O3 _vmhmm.c

_vmhmm.so: _vmhmm.o
	gcc -shared -Wa -o _vmhmm.so _vmhmm.o -lm

clean:
	rm _vmhmm.o _vmhmm.so