# Master thesis in Computer Science (Alma Mater Studiorum)

- **Title:** SRAE: Sustainable Regressive Auto Encoder
- **Author:** Riccardo Preite
- **Period:** 2021/22
- **Supervisors:** [Prof. Maurizio Gabrielli](https://www.unibo.it/sitoweb/maurizio.gabbrielli) and [Dr. Andrea Zanellati](https://www.unibo.it/sitoweb/andrea.zanellati2)
- **Abstract:** Nowadays we live in a reality characterised by economic development, innovation
technology, the improvement of the quality of life and environmental change.  The world population is facing many
problems concerning the four aspects mentioned above and, as a consequence,
sustainability has become the key point. Nevertheless, there is not a recognized and approved metric to
advise, to interested people, how to change certain aspects to grow sustainably. The United Nations have
decided to draw up a list of objectives to reach by 2030 where it is possible to find arguments in line with
what has been described so far. This collection is divided into economic, social and environmental aspects,
which are the same categories of data used for the calculation of the Sustainable Development Index. The aim of this paper is to design and develop a predictive neural network in parallel with a feedback system for a final product to be able to describe the starting context through the SDI
and/or recommend behaviors to improve the situation in a sustainable way.


# Installation

## Environment:
 
 ```console
user@pc:~$ mkdir SRAE
user@pc:~$ cd SRAE
user@pc:~/SRAE/$ git clone https://github.com/riccardopreite/SRAE_server.git
user@pc:~/SRAE/$ git clone https://github.com/riccardopreite/SRAE_model.git
```
Please check which version of torch is available for your pc.
```console
user@pc:~/SRAE/$ cd SRAE_model
user@pc:~/SRAE/SRAE_model$ pip install -r requirements.txt
user@pc:~/SRAE/SRAE_model$ cd ../SRAE_server
user@pc:~/SRAE/SRAE_server$ npm install
```

## Dataset download:

Go to [UNSDG Data portal](https://unstats.un.org/sdgs/dataportal/analytics/GlobalRegionalTrends) select all goal and download them.
```console
user@pc:~/SRAE/$ mkdir data/unsdg_dataset
```
Unzip data in **data/unsdg_dataset**.
Go to [TIME SERIES 1990-2019](https://www.sustainabledevelopmentindex.org/time-series) download and convert sdi page from xlsx file to csv. Move download file to **data/index/sdi.csv**


## Dataset preprocess
It is higly recommended to check file since some of them can have value like ">95" or "N" and it is possible that some cases are not covered in fill_na files. The problematic files will be printed during fill_sdi.py or mean_fill_sdi.py.
Fill sdi data:
```console
user@pc:~/SRAE/$ cd data/index/
user@pc:~/SRAE/data/index/$ python fill_sdi.py
user@pc:~/SRAE/data/index/$ python mean_fill_sdi.py
```
Create and fill UNSDG data:
```console
user@pc:~/SRAE/$ cd preprocess
user@pc:~/SRAE/preprocess/$ python create_indicator_csv.py
user@pc:~/SRAE/preprocess/$ python remove_nan.py
user@pc:~/SRAE/preprocess/$ python fillna.py
user@pc:~/SRAE/preprocess/$ python meanfillna.py
```
## Starting Node and Flask server

```console
user@pc:~/SRAE/$ cd SRAE_server
user@pc:~/SRAE/SRAE_server$ node server.js
```
On another console:
```console
user@pc:~/SRAE/SRAE_server$ python app.py
```

## Query example
Load both [template](https://github.com/riccardopreite/SRAE_model/blob/main/template.postman_collection.json) and [query](https://github.com/riccardopreite/SRAE_model/blob/main/query.postman_collection.json) collection json from repository in Postman.
