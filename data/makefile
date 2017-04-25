

download_and_unpack : lfw-funneled

lfw-funneled.tgz :
	curl -o lfw-funneled.tgz "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"

lfw-funneled : lfw-funneled.tgz
	tar -xzf lfw-funneled.tgz
	mv lfw_funneled lfw-funneled

.PHONY : downlaod_and_unpack
