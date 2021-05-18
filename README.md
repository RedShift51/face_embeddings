Firstly build docker container, then run it.
Build it:  sh ./build.sh
Run it:    sh ./run.sh

Inside the container go to /textask dir: cd /textask

Inside FaceDetection dir type "python setup.py install"
Inside pytorch-insightface type "python setup.py install"

Then in /textask/pipe dir type "python3 main.py --img path-to-img"

Example of preprocessing alignment:
base image:



![office5](https://user-images.githubusercontent.com/29106459/118673777-f338dc00-b801-11eb-90a1-dacedfe291c0.jpg)



Faces: before and after alignment


1)
![before0](https://user-images.githubusercontent.com/29106459/118673927-13689b00-b802-11eb-8d53-a1432597425f.png)
![after_0](https://user-images.githubusercontent.com/29106459/118673943-16fc2200-b802-11eb-868b-780591c9f75b.png)



2)
![before3](https://user-images.githubusercontent.com/29106459/118674039-29765b80-b802-11eb-81ca-1a1ff67767e3.png)
![after_3](https://user-images.githubusercontent.com/29106459/118674058-2c714c00-b802-11eb-9707-3bba32ca0728.png)



3)
![before1](https://user-images.githubusercontent.com/29106459/118674168-41e67600-b802-11eb-98b8-71f91239e378.png)
![after_1](https://user-images.githubusercontent.com/29106459/118674184-4448d000-b802-11eb-8619-ea1a9ef10970.png)
