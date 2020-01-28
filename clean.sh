#!/bin/sh


## Delete python temporary files
rm -rf python/NSSEA.egg*
rm -rf python/build
rm -rf python/dist
rm -rf python/tmp
rm -rf python/var


## Delete R temporary files
rm -f R/NSSEA/NAMESPACE
rm -f R/NSSEA/man/*.Rd
rm -f R/*.tar.gz
