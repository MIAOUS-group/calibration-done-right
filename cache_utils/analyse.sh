#!/bin/sh

NAME=`basename "$1" .txt.bz2`
echo $NAME

#bzcat $1 | awk '/^Iteration [:digit:]*[.]*/ ' > "${NAME}-iterations.txt"
#rm "${NAME}-results.csv.bz2"
#TODO forward NAME to awk script
#awk -v logname="${NAME}" -f `dirname $0`/analyse_iterations.awk < "${NAME}-iterations.txt" | bzip2 -c > "${NAME}-results.csv.bz2" # This uses system to split off awk scripts doing the analysis

bzgrep "RESULT:" "$1" | cut -b 8- | bzip2 -c > "${NAME}-results.csv.bz2"

# remove line with no data points
bzgrep -v -e "0,0,0,0,0,0,0,0,0,0$" "${NAME}-results.csv.bz2" | bzip2 -c > "${NAME}-results_lite.csv.bz2"
#paste -d"," *.csv > combined.csv
