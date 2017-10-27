FROM openjdk:8

RUN \
	apt-get update && \ 
	apt-get -y install libopenblas-dev && \ 
	apt-get -y install liblapacke-dev 

WORKDIR /datasense

ADD run.sh  			/datasense/run.sh
ADD src/main/resources  /datasense/resources/
ADD libs			  	/datasense/libs/
ADD build/libs/*	  	/datasense/libs/

ENV CP libs:resources

VOLUME [ "/datasense/data" ]

EXPOSE 8111

ENTRYPOINT [ "sh", "/datasense/run.sh" ]  
CMD [ "/datasense/data" ]

