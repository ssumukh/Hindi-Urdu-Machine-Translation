Before training data
--------------------

indictrans < input.txt -s hin -t tel -m > output1.txt	#keep output.txt for testing


Model Setup & Training & Testing
================================
	
	1. git clone https://github.com/ltrc/indic-wx-converter.git 
	(install wx-covertor, if you have skip it, beacause wx-formate preferable)
	
	2. wxconv < hin.txt -l hin -s utf > hin.wx 
	   wxconv < tel.txt -l tel -s utf > tel.wx 
	
	3. First of all ensure that you have to take clean unigram data of source and target languages and 
	    insert space between each letter of word by using regular expression.
		Eg: r A m u d u

	4. Type head and tail command in the terminal as src tgt file to check the matching 
	 	Eg: tail hin.wx tel.wx
	   	    head hin.wx tel.wx

	5. bash runGIZA++.sh source-file target-file 
		e.g: bash runGIZA++.sh hin.wx tel.wx
	
	6. above shell file will create "hin.wx-tel.wx.gmap" 

	7. python giza2tnt.py --i hin.wx-tel.wx.gmap --o hin.wx-tel.wx.tnt

	8. cut -f2 hin.wx-tel.wx.tnt | sort | uniq -c | sort -h > test.txt	
	   (lines count should not exceed more than 300, if exceed will take more time to train 
	    and this step for checking the size in "second column of .tnt file" i.e target language)
	 
	9. indictrans-trunk -d hin.wx-tel.wx.tnt  -o hin-tel -m 10 -l 3	##creating hin-tel model  
	   	(type indictrans-trunk --help for the help)	

	10. move previous hin-tel model as old-hin-tel in this path /usr/local/lib/python2.7/dist-packages/indictrans/models/ 
	   and copy the trained model 
	   	
		sudo cp -rf hin-tel /usr/local/lib/python2.7/dist-packages/indictrans/models/


How to use?
===========
After training data
--------------------

	indictrans < input.txt -s hin -t tel -m > output2.txt	#compare output1.txt output2.txt 
