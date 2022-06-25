#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:40:44 2022

@author: stasevis

function for changing dict variables via input
"""

def get_new_value(var): #, px_scale, px_scale_inv, px):
    cont = False
    while cont == False:
        var_name = input('Variable name (press enter to return): ')
        if len(var_name) == 0:
            break
        while var_name not in var.keys() and len(var_name) > 0:
            var_name = input('Variable name not recognised, please input again or press enter to return: ')

        curr_val = var[var_name]
        #if var_name in set(['r_min','r_max','height']):
            #curr_val = px_scale(curr_val,px)
        print('Currently, {0:s} = {val}'.format(var_name, val=curr_val))

        val_tmp = float(input('New value: '))
        #if var_name in set(['r_min','r_max','height']):
            #val_tmp = px_scale_inv(val_tmp,px)
        var[var_name] = val_tmp

    return var


def modify_params(var): #, px_scale, px_scale_inv, px):
    ans = input('Would you like to adjust any region parameters? [yes/no]: ')

    while ans.lower() not in set(['y','yes','n','no']):
        ans = input('Answer not recognised, please input yes(y) or no(n): ')

    if ans.lower() in set(['y','yes']):
       var = get_new_value(var) #, px_scale, px_scale_inv, px)
       final_params = False
    else:
        final_params = True

    return var, final_params