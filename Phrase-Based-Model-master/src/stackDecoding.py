# from __future__ import unicode_literals

'''this function gives the translation for a given sentence based on hypothesis recombiniation.'''
'''it takes as input the finalTranslationProbability and the input file and returns the output translation in translation.txt'''

import sys
from collections import defaultdict
import operator
import string

def findBestTranslation(finalTranslationProbability, inputFile):

	tp = defaultdict(dict)
	f=open(finalTranslationProbability,'r')
	fa = f.readlines()
	for line in fa:
		line = line.strip().split('\t')
		line[0] = line[0].translate(string.maketrans("",""), string.punctuation)
		line[1] = line[1].translate(string.maketrans("",""), string.punctuation)
		tp[line[0]][line[1]] = float(line[2])
		# print(line,'\n')
	f.close()

	# print(tp)
	
	data=[]
	f=open(inputFile,'r')
	f = f.readlines()
	for line in f:
		translationScore = defaultdict(int)
		translationSentence = defaultdict(list)
		words = line.strip().split(' ')
		for i in range(len(words)):
			words[i] = words[i].translate(string.maketrans("",""), string.punctuation)
		count = 1
		
		for i in range(len(words)):
			translation = ''

			for j in range(len(words)-count+1):
				phrase = words[j:j+count]
				phrase = ' '.join(phrase)
				if phrase in tp:
					print(phrase)
					maxx = -99999999
					var = ''
					for index,value in enumerate(tp[phrase]):
						if tp[phrase][value] > maxx:
							# print("Chosen value",value)
							maxx = tp[phrase][value]
							var = value
							# print(maxx)
					translationPhrase = var
					# print(translationPhrase)
					translationScore[count]+=tp[phrase][translationPhrase]
					translation+=translationPhrase+' '
				# else:
				# 	translation+=phrase+' '
					print(translation)
				
			if translation!='':
				translationSentence[count].append(translation)
				# print(translationSentence)
			count+=1


		maxx = -99999999
		var = ''
		for index,value in enumerate(translationScore):
			if translationScore[value] > maxx:
				# print(value)

				maxx = translationScore[value]
				var = value
			
		# print()
		# for x in translationScore.iteritems():
		# 	print(x)
		# index = max(translationScore.iteritems(), key=operator.itemgetter(1))[0]
		index = var
		finalTranslation = ' '.join(translationSentence[index])
		data.append(finalTranslation)
	f=open('translation.txt','w')
	f.write('\n'.join(data))
	f.close()

def main():
	if len(sys.argv)!=3:                                                                               #check arguments
		print("Usage :: python finalScore.py finalTranslationProbability.txt inputFile.txt ")
		sys.exit(0)

	findBestTranslation(sys.argv[1], sys.argv[2])

if __name__ == "__main__":                                                                              #main
    main()