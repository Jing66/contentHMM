#!/bin/bash
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH
for next in *.tgz
	do
		echo "Unzipping $next"
		tar -xzf $next
	done
rm *.tgz
exit 0

scp 1988/12.tgz jliu164@dp-gpu1.cs.mcgill.ca:~/corpus/nyt_corpus/data/1988

rename 's/txt\.xml/txt/g' *
ls ../../content_annotated/1998content_annotated/ | xargs mv -t ../1998content_finished 