#!/bin/sh


## Delete python temporary files
rm -rf NSSEA.egg*
rm -rf build
rm -rf dist
rm -rf tmp
rm -rf var

rm -rf tests/output/Normal/*
rm -rf tests/output/GEV/*
rm -rf tests/output/GEVMin/*

