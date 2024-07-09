sphinx-apidoc -o docs/source ../PHY399-Medical-Imaging-ROP-UTM 
sphinx-build -b html docs/source docs/build/html
sphinx-build -b latex docs/source docs/build/latex
cd docs 
make html 
make latex