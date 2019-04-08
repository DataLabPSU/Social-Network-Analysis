#normalize post data


# fin: input file
# fout: output file
# offset: offset id by this number
# eachsplit: do it for all elements in split
# skipfirst: skip first line
def normalize_uid(fin, fout, offset, eachsplit=False, skipfirst=False):
	fo = open(fout, 'w')

	with open(fin) as f:
		if skipfirst:
			next(f)

		for line in f:
			splitline = line.split(' ')
			if eachsplit:
				for i, s in enumerate(splitline):
					if int(s) == 0:
						continue
					splitline[i] = int(s) + offset
			else:
				splitline[0] = int(splitline[0]) + offset
			
			joinline = ' '.join(str(elem) for elem in splitline).replace('\n', '')
			joinline += '\n'

			fo.write(joinline)
	fo.close()


# normalize credibility- skip first line (admin)
normalize_uid(
	fin='credibility_20190405-01_07_22.txt', 
	fout='credibility_new.txt',
	offset=-121,
	eachsplit=False,
	skipfirst=True
)

# normalize edgelist- offset each element in line
normalize_uid(
	fin='edgelist_20190405-01_07_22.txt', 
	fout='edgelist_new.txt',
	offset=-121,
	eachsplit=True,
	skipfirst=False
)

# normalize impressions- skip first line (admin)
normalize_uid(
	fin='impressions_20190405-01_07_22.txt', 
	fout='impressions_new.txt',
	offset=-121,
	eachsplit=False,
	skipfirst=True
)

# normalize labellist- skip first line (admin)
normalize_uid(
	fin='labellist_20190405-01_07_22.txt', 
	fout='labellist_new.txt',
	offset=-121,
	eachsplit=False,
	skipfirst=True
)


