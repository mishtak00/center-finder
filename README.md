# center-finder

## Quick Installation

Clone the repository:
```
git clone https://github.com/mishtak00/center-finder.git
```

Create a virtual environment and install the dependencies:
```
cd center-finder
python -m venv venv
source venv/bin/activate
bash install.sh
cd run
```

Select your input mock catalog and specify number of BAO centers in it:
```
python scan.py cf_mock_catalog_83C_120R.fits 83 --full
```

Remember to deactivate the virtual environment when finished:
```
deactivate
```

### Check out the full design doc at https://docs.google.com/document/d/1AqgrDoau8i6keNp4zia9K_l3TCBOQYPmFZIa309MlE0/edit?usp=sharing
