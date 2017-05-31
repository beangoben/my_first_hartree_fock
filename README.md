## my first hartree fock
Github repo of files, a jupyter notebook and docker image that will take by the hand on building your own Hartree-Fock.

## How to use it online

Visit http://mybinder.org/ and give it the path to this github repo.
It will automatically then build all the software and provide you with a web address where you can compute and work on your first ever Hartree-Fock.

## How to use it on your computer

To run the software on any computer you need to install [docker](https://www.docker.com/).
After cloning the directory or downloading it you will need to move your terminal to
this folder and execute:

```
docker run -p 8888:8888 -v "$(pwd)":/home/jovyan/work -it "beangoben/q_solar" Hartree_Fock.ipynb
```

A browser should open with executable code.
