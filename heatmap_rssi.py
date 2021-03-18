#!/usr/bin/env python
        # -*- coding: utf-8 -*-
from numpy.core.fromnumeric import mean, shape
from time import sleep
import signal
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
import json

def handler(signum, frame):
	print("closing")
	exit(0)

signal.signal(signal.SIGINT, handler)

# FOR SCATTER PLOT HEATMAP
def scatter_dataset(x, y, weights, scattering):
	final_x = list(x)
	final_y = list(y)

	for i in reversed(range(len(weights))):
		extention_x = []
		extention_y = []

		# iterations = int((abs(min(weights))-abs(weights[i]))/len(weights))
		iterations = int((abs(min(weights))-abs(weights[i])))
		for j in range(iterations):
			extention_x.append(x[i] + round(random.uniform(-scattering, scattering), 2))
			extention_y.append(y[i] + round(random.uniform(-scattering, scattering), 2))
		final_x[i:i] = extention_x
		final_y[i:i] = extention_y

	return final_x, final_y


def filler_dataset(x, y, z, midpoint_factor):
	final_x = list(x)
	final_y = list(y)
	final_z = list(z)

	for i in reversed(range(len(x))):
		extention_x = []
		extention_y = []
		extention_z = []

		diff_x = x[i]-x[i-1]
		diff_y = y[i]-y[i-1]
		diff_z = z[i]-z[i-1]

		distance = math.hypot(diff_x, diff_y)
		angle = math.atan2(diff_y, diff_x)

		midpoints = int(distance*midpoint_factor)
		part_x = distance*math.cos(angle)/(midpoints+1)
		part_y = distance*math.sin(angle)/(midpoints+1)
		part_z = diff_z/(midpoints+1)
		for j in range(midpoints):
			midpoint_x = x[i-1] + (j+1)*part_x
			midpoint_y = y[i-1] + (j+1)*part_y
			midpoint_z = z[i-1] + (j+1)*part_z
			extention_x.append(midpoint_x)
			extention_y.append(midpoint_y)
			extention_z.append(midpoint_z)

		final_x[i:i] = extention_x
		final_y[i:i] = extention_y
		final_z[i:i] = extention_z
		if i == 1:
			break

	return final_x, final_y, final_z

# FUNCTION TO CALCULATE INTENSITY WITH QUARTIC KERNEL
def kde_quartic(d, h):
	dn = d/h
	P = (15/16.0)*(1-dn**2)**2
	return P

def heatmap(x, y, RSSIs, scattering):
	
	# GETTING X,Y MIN AND MAX
	x_min = min(x)
	x_max = max(x)
	y_min = min(y)
	y_max = max(y)

	# DEFINE GRID SIZE AND RADIUS(h)
	#PAIKSE ME GRID SIZE DYNAMIC RATIO
	spread = (x_max-x_min)
	radius = 1.5*spread/len(x)
	grid_size = radius/15

	x, y = scatter_dataset(x, y, RSSIs, scattering)

	# CONSTRUCT GRID
	x_grid = np.arange(x_min-radius, x_max+radius, grid_size)
	y_grid = np.arange(y_min-radius, y_max+radius, grid_size)
	x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

	# GRID CENTER POINT
	xc = x_mesh+(grid_size/2)
	yc = y_mesh+(grid_size/2)

	# PROCESSING
	intensity_list = []
	for j in range(len(xc)):
		intensity_row = []
		for k in range(len(xc[0])):
			kde_value_list = []
			for i in range(len(x)):
				# CALCULATE DISTANCE
				d = math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2)
				if d <= radius:
					p = kde_quartic(d, radius)
				else:
					p = 0
				kde_value_list.append(p)
			# SUM ALL INTENSITY VALUE
			p_total = sum(kde_value_list)
			intensity_row.append(p_total)
		intensity_list.append(intensity_row)

	# HEATMAP OUTPUT
	max_intensity = -1
	min_intensity = 100000
	for row in intensity_list:
		for element in row:
			if element > max_intensity:
				max_intensity = element
			if element < min_intensity:
				min_intensity = element

   	intensity = np.array(intensity_list)
	intensity = (intensity - min_intensity)/(max_intensity - min_intensity)
	intensity = intensity*(max(RSSIs)-min(RSSIs)) + min(RSSIs)

	plt.pcolormesh(x_mesh, y_mesh, intensity, cmap='gnuplot')
	# plt.plot(x, y, 'r.')
	# plt.clim(max(RSSIs),min(RSSIs))
	plt.colorbar()
	plt.title('Track RSSI heatmap s='+str(scattering))
	plt.savefig('Track_RSSI_'+str(scattering)+'.png')
	plt.show()


def linear_spectrum(x_cords, y_cords, weights):
	x_cords = np.array(x_cords)
	y_cords = np.array(y_cords)
	weights = np.array(weights)

	x, y = np.meshgrid(x_cords, y_cords)

	z = np.ones((len(weights), len(weights)))
	
	#this is for left to right
	z = np.multiply(weights, z)
	#this is for left to right
	# z = np.multiply(z,weights)

	# FIND A WAY TO DIVIDE EVERY DIAGONAL WITH A FACTOR
	# z.flat[::len(weights)] /= 1
	# x and y are bounds, so z should be the value *inside* those bounds.
	# Therefore, remove the last value from the z array.
	# z = z[:-1, :-1]

	z_min, z_max = z.min(), z.max()

	fig, ax = plt.subplots()

	c = ax.pcolormesh(x, y, z, cmap='gnuplot', vmin=z_min, vmax=z_max)
	# this is the original but has problems
	# c = ax.contourf(x, y, z, 15, cmap='gnuplot', vmin=z_min, vmax=z_max)
	ax.set_title('Linear RSSI spectrum')
	# set the limits of the plot to the limits of the data
	ax.axis([x.min(), x.max(), y.min(), y.max()])
	fig.colorbar(c, ax=ax)

	plt.savefig('Linear_RSSI.png')
	plt.show()

# ---------------------------------

def clear_tuples_bias(tuple_array, axis):
	# print(tuple_array)
	array = []
	if axis is "x":
		axis = 0
	elif axis is "y":
		axis = 1
	else:
		axis = 2

	for i in range(len(tuple_array)):
		array.append(tuple_array[i][axis])

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
	
	print(indecies_array)
	indecies_array_reverse = indecies_array[::-1]
	cleared_array = tuple_array
	
	for i in indecies_array_reverse:
		cleared_array.pop(i)

	return cleared_array

def read_json(filename):
	x = []
	y = []
	rssi = []
	with open(filename, "r") as read_file:
		data = json.load(read_file)
		for line in data:
			_x = line['x']
			_y = line['y']
			_link = line['link']

			if _x == 0.0 and _y == 0.0 and _link == 0.0:
				continue

			x.append(_x)
			y.append(_y)
			rssi.append(_link)

	x, y, rssi = filler_dataset(x, y, rssi, 3)
	return x, y, rssi

def compare_items_x(a, b):
	axis = 0
	if a[axis] > b[axis]:
		return 1
	elif a[axis] == b[axis]:
		return 0
	else:
		return -1

def compare_items_y(a, b):
	axis = 1
	if a[axis] > b[axis]:
		return 1
	elif a[axis] == b[axis]:
		return 0
	else:
		return -1

def read_sorted_json(filename,axis):
	x = []
	y = []
	rssi = []
	tuples = []
	with open(filename, "r") as read_file:
		data = json.load(read_file)
		for line in data:
			_x = line['x']
			_y = line['y']
			_link = line['link']
			if _x == 0.0 and _y == 0.0 and _link ==0.0:
				continue

			t = _x,_y,_link
			tuples.append(t)

	# tuples = clear_tuples_bias(tuples,"x")
	tuples = clear_tuples_bias(tuples,"y")

	if axis is "x":
		tuples.sort(compare_items_x)
	if axis is "y":
		tuples.sort(compare_items_y)

	for t in tuples:
		x.append(t[0])
		y.append(t[1])
		rssi.append(t[2])


	x, y, rssi = filler_dataset(x,y,rssi, 3)
	return x, y, rssi

# ---------------------------------

filename = "wifi_strength1.json"
x, y, RSSIs = read_sorted_json(filename,"x")
heatmap(x,y,RSSIs,0.05)
linear_spectrum(x,y,RSSIs)
