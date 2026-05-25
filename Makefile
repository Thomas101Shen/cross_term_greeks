.PHONY: all cpp test clean format format-check python-test wheel latex latex-clean

all: cpp

cpp:
	$(MAKE) -C cpp all

test:
	$(MAKE) -C cpp test

clean:
	$(MAKE) -C cpp clean

format:
	$(MAKE) -C cpp format

format-check:
	$(MAKE) -C cpp format-check

python-test:
	.venv/bin/python -m unittest discover -s tests -v

wheel:
	.venv/bin/python setup.py bdist_wheel

latex:
	$(MAKE) -C write_up all

latex-clean:
	$(MAKE) -C write_up clean
