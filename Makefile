all: markov.so

markov.so: markov.c
	python setup.py build
	cp build/lib.linux-*/markov.so $@

clean:
	rm -fr markov.so build

