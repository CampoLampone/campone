# campone
Python project used for Jetbots on this year's Campo Lampone.

Directory `campone` contains a module containing shared code and code not meant to be majorly adjusted by the participants.

Before starting `main.py` (or any other code) you need to install `campone` and requirements.

To install `campone` you can run:
```bash
pip install -r requirements.txt
pip install -e .
```
The `-e` argument makes it an "editable" module, that means it can be installed and imported from anywhere but still be edited.
Installing this package in this way makes it very convenient, because you don't need to do path hacks to import the shared code.

## The "architecture"
`main.py` initializes the camera thread and then starts all of the other workers.

The workers are meant mainly as a way to process the images using a few different methods at the same time without killing the performance.

Tests for some Jetbot functionality are included but are not very sofisticated.
