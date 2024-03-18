# Apps similarity via embeddings

This command line tool computes the embeddings of an App and finds the best match from previously calculated embeddings of Apps.
The model used for creation of the embeddings is [CodeT5+](https://huggingface.co/Salesforce/codet5p-110m-embedding)

## Setup

Using Python 3.10
* Create a virtual environment
```
python -m venv /path/to/new/virtual/environment
```
* Install requirements.txt
```
pip install -r requirements.txt
```

## Usage

Running
```
python embeddings.py -g /path/to/file
```
will generate the embeddings from *dataset.json* and save them to *embeddings.pickle*.
If the **-g** is not specified the program will assume the file *embeddings.pickle* is already generated.
For testing purposes *test_app.json* could be used

## Tests

The tests can be run with
```
python -m unittest test_embeddings.py
```