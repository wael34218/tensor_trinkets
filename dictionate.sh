#!/bin/bash

# TODO: Normalize numbers, dates and other stuff
cat $1 | sed 's/[[:punct:]]//g' | tr '[:upper:]' '[:lower:]' > $1.clean
cat $1.clean | tr " " "\n" | sed '/^\s*$/d' | sort | uniq -c | sort -nr | awk 'BEGIN{i=5}{printf("%s\t%s\n",i,$2);i+=1}' > tmp
echo -e '1\t<eos>\n2\t<pad>\n3\t<unk>\n4\t<go>' | cat - tmp > $1.dic

rm tmp
