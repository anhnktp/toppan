# Prepare to run

Firstly, create working directory named ***project5/*** in directory /home/awl_exactreid. 

Then, you need to download file ***models.zip*** from *https://drive.google.com/file/d/1ilSMbkAO51D2LijprbcGkTFkAUh5Tl49/view* to folder ***/home/awl_exactreid/project5*** and unzip. Create dir named cropped_images

(If dir ***./project5/*** and **./project5/models/** already exist, you can skip this step)
~~~bash
cd /home/awl_exactreid
mkdir project5 && cd project5
mkdir cropped_images
unzip models.zip && rm -rf models.zip
~~~
 
# Run flowing line to run AI Engine 
If you want to flush database before run engine
~~~bash
$ sudo rm -rf data/
~~~
Run engine
~~~bash
# Adds docker to X server access control list (ACL)
$ xhost + local:docker

# Build and run the services
$ docker-compose up 
 ~~~