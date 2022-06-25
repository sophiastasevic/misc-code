#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 00:48:46 2022

@author: stasevis
"""

import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.coordinates as coord
import numpy as np
data_epoch='2015-04-12'
star_data = dict(ra=189.94228138007,
                 ra_err=0.0237, #mas
                 dec=-49.19883071290,
                 dec_err=0.0179, #mas
                 pmra=-29.847,
                 pmra_err=0.028,
                 pmdec=-14.979,
                 pmdec_err=0.020,
                 ref_epoch=2016.0,
                 plx=7.6878,
                 plx_err=0.0314)

cc_data = dict(ra_sep=2.3021817, #arcsec
               dec_sep=-5.744104625) #arcsec

gaia_cc_data = dict(ra_sep=3.5344528560358413,
               dec_sep=-5.763055680009188,
               ra=189.94326317253,
               ra_err=0.7627, #mas
               dec=-49.20043156170,
               dec_err=0.4590, #mas
               plx=1)

c_star=coord.SkyCoord(ra=star_data['ra'] * u.deg,
            dec=star_data['dec'] * u.deg, 
            distance=Distance(parallax=star_data['plx'] * u.mas),
            pm_ra_cosdec=star_data['pmra'] * u.mas/u.yr,
            pm_dec=star_data['pmdec'] * u.mas/u.yr,
            obstime=Time(star_data['ref_epoch'], format='jyear',
            scale='tcb'))

c_gaia_cc=coord.SkyCoord(ra=gaia_cc_data['ra'] * u.deg,
            dec=gaia_cc_data['dec'] * u.deg, 
            distance=Distance(parallax=gaia_cc_data['plx'] * u.mas),
            pm_ra_cosdec=star_data['pmra'] * u.mas/u.yr,
            pm_dec=star_data['pmdec'] * u.mas/u.yr,
            obstime=Time(star_data['ref_epoch'], format='jyear',
            scale='tcb'))

#mot=c_star.apply_space_motion(dt=10. * u.year)
star_mot=c_star.apply_space_motion(new_obstime=Time(data_epoch))
gaia_cc_mot=c_gaia_cc.apply_space_motion(new_obstime=Time(data_epoch))

pa_2016=c_star.position_angle(c_gaia_cc).to(u.deg)  
sep_2016=c_star.separation(c_gaia_cc)
offset_2016= c_star.directional_offset_by(pa_2016,sep_2016).to(u.arcsecond)

pa_obs=star_mot.position_angle(gaia_cc_mot).to(u.deg)  
sep_obs=star_mot.separation(gaia_cc_mot)
offset_obs= star_mot.directional_offset_by(pa_obs,sep_obs).to(u.arcsecond)
