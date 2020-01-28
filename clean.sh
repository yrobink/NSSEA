#!/bin/sh


## Delete python temporary files
rm -rf python/NSSEA.egg*
rm -rf python/build
rm -rf python/dist
rm -rf python/tmp
rm -rf python/var

rm -rf tests/output/Normal/*
rm -rf tests/output/GEV/*


## Delete R temporary files
rm -f R/NSSEA/NAMESPACE
rm -f R/NSSEA/man/*.Rd
rm -f R/*.tar.gz
