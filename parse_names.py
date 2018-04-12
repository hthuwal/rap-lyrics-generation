f = open('names.txt','r')
mappin = {}
artist = {}
for l in f:
	_,n=l.rstrip().split()
	a = n.split('-')[1]
	arti = n.split('-')[0]
	artist[a]=arti
	if a in mappin: mappin[a].append(n.split('-')[2])
	else : mappin[a]=[n.split('-')[2]]
for m in mappin:
	print(m,len(mappin[m]),artist[m])
	for i in mappin[m]:
		print('\t- ',i)

