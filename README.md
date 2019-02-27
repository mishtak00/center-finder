# center-finder

## Installation

Clone the repository:
```
git clone https://github.com/yujie-liu/center-finder.git
```

Create a virtual environment and install the dependencies:
```
cd center-finder
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
 
Run an example test:
```
python tests/test_blob.py
```

Or customize input mock catalog and number of generated centers in the mock:
```
python tests/blob_example.py data/cf_mock_catalog_83C_120R.fits 83
```

Remember to deactivate the virtual environment when finished:
```
deactivate
```
