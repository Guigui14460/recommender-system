# Recommender system

University project proposed by our professors as part of an annual project (starting and ending in March) of our 3rd year in computer science degree at the University of Caen Normandy. The goal of this subject is to make us discover how we work in real life in computer science (communication with a customer, project development, etc).

## Table of contents

  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Commands](#commands)
  - [Authors](#authors)
  - [License](#license)

## Introduction
The goal of this project is to realize a system of object recommendation (of movies in our case). So we had to get the data we needed, search/use/improve existing recommendation algorithms (such as [TF-iDF](https://www.wikiwand.com/en/Tf%E2%80%93idf)) in order to better target users' tastes and desires.

## Setup
You need to have Python 3 installed on your machine to be able to use this project.

Launch this command (at the root of the project directory) to install packages :
```shell
$ pipenv install
```

After, download the movie dataset [here](https://www.kaggle.com/rounakbanik/the-movies-dataset) and put all `.csv` files in a `data` folder at the root of the project.

## Commands
- To enter in the virtual environment :
```shell
$ pipenv shell
```

- To launch a Python shell inside the virtual environment :
```shell
(test-rnrvitb) $ python
```

- To quit the virtual environment :
  - Under Unix system :
  ```shell
  $ deactivate
  ```
  - Under Windows system :
  ```powershell
  $ exit
  ```

- To remove the virtual environment :
```shell
$ pipenv --rm
```

## Authors
- [KASSA Rina](https://github.com/rinakassa7)
- [OUNESLI Melissa](https://github.com/Melissa-Ou)
- [URAZOV Zhandos](https://github.com/zhandu)
- [LETELLIER Guillaume](https://github.com/Guigui14460)

## License
Project under the GPLv3 license.
