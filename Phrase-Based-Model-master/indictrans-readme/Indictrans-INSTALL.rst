Installation
============

Dependencies
^^^^^^^^^^^^
	pbr
	six
	future
	cython>=0.24.0a0
	numpy>=1.10.4
	scipy>=0.13.3

Install
^^^^^^^

    Change to the cloned directory:
        cd indic-trans
        pip install -r requirements.txt
	
	if dependencies packages not installing properly through pip, download dependencies packages from the web and install individually.

        python setup.py build
        python setup.py install

	
After successfully installed, try this command to test in terminal 

	indictrans < test.txt -s hin -t tel -m > output.txt 
