# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:28:22 2022

@author: chomeza
"""

def get_star_info(cube_name):
    master_cube_fits = F.open(cube_name)
    hdr = master_cube_fits[0].header
    
    time = Time(hdr['DATE-OBS'])
    print(time)
    coords = coord.SkyCoord(hdr['RA']*u.degree,hdr['DEC']*u.degree)
    print(coords)
    name_star = hdr['OBJECT']
    
    Query = query_simbad(time,coords,name=name_star,debug=True)
    
    master_cube_fits.close()
    
    return Query

def query_simbad(date,coords,name=None,debug=True,limit_G_mag=15):
    """
    Function that tries to query Simbad to find the object. 
    It first tries to see if the star name (optional argument) is resolved 
    by Simbad. If not it searches for the pointed position (ra and
    dec) in a cone of radius 10 arcsec. If more than a star is detected, it 
    takes the closest from the (ra,dec).
    Input:
        - date: an astropy.time.Time object (e.g. date = Time(header['DATE-OBS'])
        - name: a string with the name of the source.
        - coords: a SkyCoord object. For instance, if we extract the keywords 
            of the fits files, we should use
            coords = coord.SkyCoord(header['RA']*u.degree,header['DEC']*u.degree)
            coord.SkyCoord('03h32m55.84496s -09d27m2.7312s', ICRS)
    Output:
        - a dictionary with the most interesting simbad keywords.
    """
    search_radius = 10*u.arcsec # we search in a 10arcsec circle.
    search_radius_alt = 210*u.arcsec # in case nothing is found, we enlarge the search
        # we use 210 arcsec because Barnard star (higher PM star moves by 10arcsec/yr --> 210 arcsec in 21yrs)
    customSimbad = Simbad()
    customSimbad.add_votable_fields('parallax','plx_error','flux(V)','flux(R)','flux(G)','flux(I)','flux(J)','flux(H)',\
                                    'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                                   'propermotions','ra(2;A;ICRS;J2000;2000)',\
                                 'dec(2;D;ICRS;J2000;2000)')
    # First we do a cone search around he coordinates
    search = customSimbad.query_region(coords,radius=search_radius)
    
    if search is None and name is None:
        # If the cone search failed and no name is provided we cannot do anything more
        print('No star identified for the RA/DEC pointing. Enlarging the search to {0:.0f} arcsec'.format(search_radius_alt.value))
        search = customSimbad.query_region(coords,radius=search_radius_alt)
        if search is None:
            print('No star identified for the RA/DEC pointing. Stopping the search.')
            return None
        else:
            validSearch = search[search['FLUX_G']<limit_G_mag]
            nb_stars = len(validSearch)                
        
    elif search is None and name is not None:
        # If the cone search failed but a name is provided, we query that name
        print('No star identified within {0:.0f} arcsec of the RA/DEC pointing. Querying the target name {1:s}'.format(search_radius.to(u.arcsec).value,name))
        # get the star from target name
        simbad_dico = get_dico_star_properties_from_simbad_target_name_search(name,customSimbad)
        if 'simbad_FLUX_V' in simbad_dico.keys():  
            nb_stars = -1 # nothing else to be done! 
            print('Star {0:s} identified using the target name'.format(simbad_dico['simbad_MAIN_ID']))
        else:
            print('No star corresponding to the target name {0:s}. Enlarging the search to {1:.0f} arcsec'.format(name,search_radius_alt.value))
            search = customSimbad.query_region(coords,radius=search_radius_alt)
            if search is None:
                print('No star identified for the RA/DEC pointing. Stopping the search.')
                return None
            else:
                validSearch = search[search['FLUX_G']<limit_G_mag]
                nb_stars = len(validSearch)                
    else:
        # If the cone search returned some results, we count the valid candidates.
        nb_stars = len(search)
        validSearch = search[search['FLUX_G']<limit_G_mag]
        nb_stars = len(validSearch)    
        
    if nb_stars==0:
        print('No star identified for the pointing position. Querying the target name')
        # get the star from target name if we have it in the text file.
        simbad_dico = get_dico_star_properties_from_simbad_target_name_search(name,customSimbad)
        # if we found a star, we add the distance between ICRS coordinates and pointing
        if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
            coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'],simbad_dico['simbad_DEC_ICRS']])
            coords_ICRS = coord.SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
            sep_pointing_ICRS = coords.separation(coords_ICRS).to(u.arcsec).value
            simbad_dico['simbad_separation_RADEC_ICRSJ2000']=sep_pointing_ICRS
        # if we found a star, we add the distance between Simbad current coordinates and pointing
        if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
            coords_current_str = ' '.join([simbad_dico['simbad_RA_current'],simbad_dico['simbad_DEC_current']])
            coords_current = coord.SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
            sep_pointing_current = coords.separation(coords_current).to(u.arcsec).value
            simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
    elif nb_stars>0:
        if nb_stars ==1:
            i_min=0
            print('One star found: {0:s} with G={1:.1f}'.format(\
                  validSearch['MAIN_ID'][i_min],validSearch['FLUX_G'][i_min]))
        else:
            print('{0:d} stars identified within {1:.0f} or {2:.0f} arcsec. Querying the target name'.format(nb_stars,search_radius.value,search_radius_alt.value)) 
            # First we query the target name
            simbad_dico = get_dico_star_properties_from_simbad_target_name_search(name,customSimbad)
            if ('simbad_MAIN_ID' in simbad_dico):
                # the star was resolved and we assume there is a single object corresponding to the search 
                i_min=0
            else:
                print('Target not resolved or not in the list. Selecting the closest star.')
                sep_list = []
                for key in validSearch.keys():
                    if key.startswith('RA_2_A_FK5_'):
                        key_ra_current_epoch = key
                    elif key.startswith('DEC_2_D_FK5_'):
                        key_dec_current_epoch = key
                for i in range(nb_stars):
                    ra_i = validSearch[key_ra_current_epoch][i]
                    dec_i = validSearch[key_dec_current_epoch][i]
                    coord_str = ' '.join([ra_i,dec_i])
                    coords_i = coord.SkyCoord(coord_str,frame=FK5,unit=(u.hourangle,u.deg))
                    sep_list.append(coords.separation(coords_i).to(u.arcsec).value)
                i_min = np.argmin(sep_list)
                min_sep = np.min(sep_list)
                print('The closest star is: {0:s} with G={1:.1f} at {2:.2f} arcsec'.format(\
                  validSearch['MAIN_ID'][i_min],validSearch['FLUX_G'][i_min],min_sep))
        simbad_dico = populate_simbad_dico(validSearch,i_min)
    simbad_dico['DEC'] = coords.dec.to_string(unit=u.degree,sep=' ')
    simbad_dico['RA'] = coords.ra.to_string(unit=u.hourangle,sep=' ')
    # if we found a star, we add the distance between ICRS coordinates and pointing
    if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
        coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'],simbad_dico['simbad_DEC_ICRS']])
        coords_ICRS = coord.SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
        sep_pointing_ICRS = coords.separation(coords_ICRS).to(u.arcsec).value
        simbad_dico['simbad_separation_RADEC_ICRSJ2000']=sep_pointing_ICRS
    # if we found a star, we add the distance between Simbad current coordinates and pointing
    if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
        coords_current_str = ' '.join([simbad_dico['simbad_RA_current'],simbad_dico['simbad_DEC_current']])
        coords_current = coord.SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
        sep_pointing_current = coords.separation(coords_current).to(u.arcsec).value
        simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
        print('Distance between the current star position and pointing position: {0:.1f}arcsec'.format(sep_pointing_current))
    # if we found a star with no R magnitude but with known V mag and spectral type, we compute the R mag.
    # if 'simbad_FLUX_V' in simbad_dico.keys() and 'simbad_SP_TYPE' in simbad_dico.keys() and 'simbad_FLUX_R' not in simbad_dico.keys():
    #     color_VminusR = color(simbad_dico['simbad_SP_TYPE'],filt='V-R')
    #     if np.isfinite(color_VminusR) and np.isfinite(simbad_dico['simbad_FLUX_V']):
    #         simbad_dico['simbad_FLUX_R'] = simbad_dico['simbad_FLUX_V'] - color_VminusR
    return simbad_dico

def get_dico_star_properties_from_simbad_target_name_search(name,customSimbad):
    """
    Method not supposed to be used outside the query_simbad method
    Returns a dictionary with the properties of the star, after querying simbad
    using the target name
    If no star is found returns an empty dictionnary
    """
    simbad_dico = {}
    simbadsearch = customSimbad.query_object(name)
    if simbadsearch is None:
        # failure
        return simbad_dico
    else:
        # successful search
        return populate_simbad_dico(simbadsearch,0)

def populate_simbad_dico(simbad_search_list,i):
    """
    Method not supposed to be used outside the query_simbad method
    Given the result of a simbad query (list of simbad objects), and the index of 
    the object to pick, creates a dictionary with the entries needed.
    """    
    simbad_dico = {}
    #print(simbad_search_list.keys())
    for key in simbad_search_list.keys():
        
        if key in ['MAIN_ID','SP_TYPE','ID_HD','OTYPE','OTYPE_V','OTYPE_3']: #strings
            if not simbad_search_list[key].mask[i]:
                simbad_dico['simbad_'+key] = simbad_search_list[key][i]
        elif key in ['PLX_VALUE','PLX_ERROR','FLUX_V', 'FLUX_R', 'FLUX_G','FLUX_I', 'FLUX_J', 'FLUX_H', 'FLUX_K','PMDEC','PMRA']: #floats
            if not simbad_search_list[key].mask[i]:
                simbad_dico['simbad_'+key] = float(simbad_search_list[key][i])
        elif key.startswith('RA_2_A_FK5_'): 
            simbad_dico['simbad_RA_current'] = simbad_search_list[key][i]      
        elif key.startswith('DEC_2_D_FK5_'): 
            simbad_dico['simbad_DEC_current'] = simbad_search_list[key][i]
        elif key=='RA':
            simbad_dico['simbad_RA_ICRS'] = simbad_search_list[key][i]
        elif key=='DEC':
            simbad_dico['simbad_DEC_ICRS'] = simbad_search_list[key][i]     
    return simbad_dico



# Simbad Query

query = get_star_info(master_cube)

plx_star = query['simbad_PLX_VALUE']
pm_ra = query['simbad_PMRA']
pm_dec = query['simbad_PMDEC']
Spectral_type = query['simbad_SP_TYPE']