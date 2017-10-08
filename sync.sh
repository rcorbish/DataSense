#!/bin/sh

#set -x 

PROJECT=$( basename $(pwd) )

if [ "$1" = "get" ]
then
	echo "Getting from desktop"
	rsync -r mars:workspace/${PROJECT} ${HOME}/workspace
elif [ "$1" = "put" ]
then
	echo "Sending to desktop"
	rsync -r ${HOME}/workspace/${PROJECT} mars:workspace
else
	echo "usage $0 [get|put]"
	exit 1
fi

