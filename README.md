Firstly build docker container, then run it.
Build it:  sh ./build.sh
Run it:    sh ./run.sh

Inside the container go to /textask dir: cd /textask

Inside FaceDetection dir type "python setup.py install"
Inside pytorch-insightface type "python setup.py install"

Then in /textask/pipe dir type "python3 main.py --img path-to-img"
