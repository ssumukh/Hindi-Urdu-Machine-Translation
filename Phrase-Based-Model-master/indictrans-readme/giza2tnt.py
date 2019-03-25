import re
import sys
import argparse

def align_mappings(INFILE, OUTFILE):
    """aligns mapped words of GIZA++ output"""
    align_pair = -1
    for i, line in enumerate(INFILE):
	trailing = list()
        if i%3 == 0:
            align_pair += 1
	if i%3 == 1:
	    # line 2 contains target sentence
	    target = line.split()
	elif i%3 == 2:
	    # line 3 contains source word alignments with target words
	    source = re.findall(r"\s*(.*?)\s*\({\s*(.*?)\s*}\)",line)
	    source = [list(tpl) for tpl in source]
	    unalighned = source.pop(0)[1]
	    first = 1
	    if unalighned:
		unalighned = unalighned.split()#[::-1]
		for val in unalighned:
		    trailing.append(val)
		    flag = False
		    for i_,maping in enumerate(source):
			pos = maping[1].split()
			#if pos != sorted(pos, key=int):
			    #print >>sys.stderr, 'Wrong assumption'
			for p,v in enumerate(pos):
			    if int(val) == first:
				first += 1
				trailing = trailing[:-1]
				pos.insert(-1, val)  
				source[i_][1] = ' '.join(pos)
				flag = True
				break
			    elif str(int(val)-1) == v:
				trailing = trailing[:-1]
				pos.append(val)
				source[i_][1] = ' '.join(pos)
				flag = True
				break
			if flag: break
	    if trailing: source[-1][1] += ' ' + ' '.join(trailing[::-1])
            order = reduce(lambda x,y: x+y, [map(int, x[1].split()) for x in source])
            if sorted(order) != order:
                #sys.stderr.write('%d\n' %align_pair)
	        for word, idx in source:
		    sys.stderr.write('%s\n' %(word+'\t'+''.join([target[int(j)-1] \
			    for j in sorted(idx.split(), key=int)]) if idx else word+'\t'+'_'))
	        sys.stderr.write('\n')
            else:
	        for word, idx in source:
		    OUTFILE.write('%s\n' %(word+'\t'+''.join([target[int(j)-1] \
			    for j in sorted(idx.split(), key=int)]) if idx else word+'\t'+'_'))
	        OUTFILE.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="hin2urd_mapper")
    parser.add_argument('--i', metavar='input', dest="INFILE", type=argparse.FileType('r'), required=True, help="<input-file>")
    parser.add_argument('--o', metavar='output', dest="OUTFILE", type=argparse.FileType('w'), default=sys.stdout, help="<output-file>")
    args = parser.parse_args()

    align_mappings(args.INFILE, args.OUTFILE)
