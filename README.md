## my first hartree fock

Ever wanted to program your first Hartree-fock? The building block for most algorithm in Quantum Chemistry?

Then this Github repo if for you. It is a jupyter notebook (**index.ipynb**) and a -docker image](https://hub.docker.com/r/beangoben/my_first_hartree_fock/) that will take you by the hand (mostly) on building your own Hartree-Fock.

## How to use it on your computer (recommended)

To run the software on any computer you need to install [docker](https://www.docker.com/).

After cloning the directory or downloading it you will need to move your terminal to
this folder and execute:

```
docker run -p 8888:8888 -v "$(pwd)":/home/jovyan/work -it "beangoben/my_first_hartree_fock" index.ipynb
```

A browser should open with executable code, if not visit [http://localhost:8888/](http://localhost:8888/).

## How to install locally

This program uses:

* Python 2.7
* Pyquante 1.6.5
* Numpy, Scipy, Matplotlib
* imolecule
* py3dmol
* jupyter notebook

## How to use it online

Visit http://mybinder.org/ and give it the path to this github repo.
It will automatically then build all the software and provide you with a web address where you can compute and work,

Careful, work is temporary so you might want to download your once in a while.
