#!/usr/bin/env python
        # -*- coding: utf-8 -*-
from numpy.core.fromnumeric import shape
from pluto.pluto_sdr import PlutoSdr
from time import sleep
import signal

import math
import numpy as np

sdr = PlutoSdr()
print(sdr.name)
sdr.rx_lo_freq = 6000
sdr.tx_state = sdr.TX_OFF
sdr._set_rxBW(5)
sdr.rxStatus()
#print(sdr.phy.channels[4].attrs)

def handler(signum, frame):
	print("closing")
	exit(0)

signal.signal(signal.SIGINT, handler)

def get_rssi():
	mean_rssi_array = []
	
	for j in range(0,5):
		rssi_array = []
		
		for i in range(0,100):
			rssi_str = sdr.rssi
			rssi_float = float(rssi_str.split(" ")[0])

			rssi_array.append(rssi_float)
		
		rssi_array = clear_bias(rssi_array)
		mean_rssi = np.mean(rssi_array)

		mean_rssi_array.append(round(mean_rssi, 2))

	mean_rssi_array = clear_bias(mean_rssi_array)

	mean_of_means = np.mean(mean_rssi_array)
	return round(mean_of_means, 3)

def get_rssi_simple():
	rssi_array = []
		
	for i in range(0,500):
		rssi_str = sdr.rssi
		rssi_float = float(rssi_str.split(" ")[0])

		rssi_array.append(rssi_float)
	
	rssi_array = clear_bias(rssi_array)
	mean_rssi = np.mean(rssi_array)

	return round(mean_rssi, 3)

def clear_bias(array):
	max = np.max(array)
	min = np.min(array)
	mean = np.mean(array)

	spread = max - min

	thresh_low = mean - spread/2
	thresh_high = mean + spread/2
	
	indecies_array = []
	for i in range(len(array)):
		if not ((array[i] > thresh_low) and (array[i] < thresh_high)):
			indecies_array.append(i)
	
	if len(array)==len(indecies_array):
		return array
	
	indecies_array_reverse = indecies_array[::-1]
	cleared_array = array
	
	for i in indecies_array_reverse:
		cleared_array.pop(i)

	return cleared_array

def print_rssi():
	while True:
		print(get_rssi())
		sleep(0.1)


#---------------------------------


print_rssi()
