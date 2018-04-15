#!/bin/bash
cd analyze
java -cp ".:./copylibstask.jar:./weka.jar" rhymeapp.mainUI $1
cd ..
