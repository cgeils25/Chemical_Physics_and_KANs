"""
Tests imports of required packages from requirements.txt:

- jupyter
- matplotlib
- numpy
- pandas
- pytest
- torch
- rdkit
- sklearn
- seaborn
- sympy
- tqdm
- kan
"""

required_packages = ['jupyter', 'matplotlib', 'numpy', 'pandas', 'pytest', 'torch', 'rdkit', 'sklearn', 'seaborn', 'sympy', 'tqdm', 'kan', 'polars']
# man this would be so much easier if the name of a package was the same as the name of the module you import from it

def test_imports():
    # test that all required packages can be imported
    for package in required_packages:
        assert __import__(package) is not None
