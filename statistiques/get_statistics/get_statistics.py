# -*- coding: utf-8 -*-
"""create statistics about the annotated images 
Created on Wed Aug 12 18:05:10 2020

Usage:
    python get_statistics.py --dir [annotation directory path]

Author:
    Ahmed Haj Yahmed
"""
import os
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import argparse

def get_nb_elements(objects):
    count_personne , count_vehicule = 0, 0
    for ob in objects:
        if ob[0].text == "personne":
            count_personne += 1
        else:
            count_vehicule += 1
    return count_personne, count_vehicule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='annotation directory path')

    args = vars(parser.parse_args())
    
    if not(os.path.isdir('./results')):
        os.mkdir('./results')

    with open('./results/statistique.csv', 'w', newline='') as file:
        writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
        writer.writerow(["nom_image", "nombre_personne", "nombre_vehicule", "niveau_altitude", "path"])


    all_files = os.listdir(args['dir'])
    
    all_personne, all_vehicule = 0, 0
    nb_file_only_personne, nb_file_only_vehicule = 0, 0
    
    for file in all_files:
        tree = ET.parse(os.path.join(args['dir'],file))
        root = tree.getroot()
        nom_image = root[1].text
        nb_personne, nb_vehicule = get_nb_elements(root.findall("./object"))
        all_personne += nb_personne
        all_vehicule += nb_vehicule
        if nb_vehicule == 0:
            nb_file_only_personne += 1
        elif nb_personne ==0:
            nb_file_only_vehicule += 1
        path = root[2].text
        with open('./results/statistique.csv', 'a', newline='') as file:
            writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerow([nom_image, nb_personne, nb_vehicule,'' , path])
    
    with open('./results/statistique.csv', 'a', newline='') as file:
            writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerow([''])
            writer.writerow(['totale personne : ', all_personne])
            writer.writerow(['totale vehicule : ', all_vehicule])
            writer.writerow(['image contenant seulement des personnes : ', nb_file_only_personne])
            writer.writerow(['image contenant seulement des vehicules : ', nb_file_only_vehicule])
    
    read_file = pd.read_csv('./results/statistique.csv')
    read_file.to_excel('./results/statistique.xlsx', index=None, header=True)
    os.remove('./results/statistique.csv')  
    
if __name__ == '__main__':
    main()