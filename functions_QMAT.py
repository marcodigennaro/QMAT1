#!/scratch/snx3000/mdigenna/anaconda2/bin/python

import subprocess
from QMAT import path2logs

def check_running_jobs():
    cmd     = "squeue -v -u mdigenna | grep mdigenna | awk '{print $4}' | uniq "
    running = subprocess.Popen( cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True )
    running_list = running.stdout.read().split('\n')
    #running = subprocess.getoutput( cmd ) #, shell=True, stdout=subprocess.PIPE)
    #running_list = [ elem for elem in running.split('\n') if not elem.startswith('squeue')  ] 
    return( running_list )

def get_lowest_state( code, data_list ):
    all_items   = [ item[1] for item in data_list if list(set(item[1]))!=['nan'] ]
    if len( all_items ) == 0 :
       results_inst  = 'nan'
       results_list  = ['nan', 'nan', 'nan']
       elem_log_line = '{} no results\n'.format(4*' ')
    else:
       all_items.sort( key=lambda x: x[2] )
       results_list  = all_items[0]
       results_inst  = [ item[0] for item in data_list if item[1]== results_list ][0]
       elem_log_line = '{} {} lowest energy instance = {} ( en = {}, vol = {}, press = {} )\n'.format(4*' ', code[0:2], results_inst,  *results_list )
    return( results_inst, results_list, elem_log_line )

def print_relative_error( pw_1atom_en, abi_1atom_en, lowest_energy_dict, kpt_density ):
    lowest_pw_en  = 0
    lowest_abi_en = 0
    for key, val in lowest_energy_dict.items():
        if float( val[0][1][1] ) < lowest_pw_en:
           lowest_pw_en     = val[0][1][1]
           lowest_pw_state  = ( key, val[0] )
        if float( val[1][1][1] ) < lowest_abi_en:
           lowest_abi_en    = val[1][1][1]
           lowest_abi_state = ( key, val[1] )
    print( 'lowest pw  state = {}, inst. = {}: E = {}, V = {}'.format( lowest_pw_state[0],  lowest_pw_state[1][0] , lowest_pw_state[1][1][1] , lowest_pw_state[1][1][0]  ))
    print( 'lowest abi state = {}, inst. = {}: E = {}, V = {}'.format( lowest_abi_state[0], lowest_abi_state[1][0], lowest_abi_state[1][1][1], lowest_abi_state[1][1][0]  ))

    VOL_LOG = open( '{}/Volume_error.dat'.format( path2logs ) , 'w+')
    FE_LOG  = open( '{}/Formation_energy.dat'.format( path2logs ) , 'w+')
    VOL_LOG.write( '##{:5s}\t{:10s}\t{:10s}\t{:10s}\n'.format( 'spgr', 'VOL(PW)','VOL(ABI)','(V(PW)-V(ABI))/V(PW)'))
    FE_LOG.write(  '##{:5s}\t{:10s}\t{:10s}\t{:10s}\n'.format( 'spgr', 'FE(PW)' ,'FE(ABI)' ,'(FE(PW)-FE(ABI))/FE(PW)'))
#    for spgr in spgr_list:
#        try:
#          val = lowest_energy_dict[ 'spgr_{}'.format(spgr) ]
#          pw_inst , [pw_vol , pw_en,  pw_press ] = val[0]
#          abi_inst, [abi_vol, abi_en, abi_press] = val[1]
#          VOL_LOG.write( '{}\t{:10s}\t{:10s}\t{}\n'.format(spgr, str(pw_vol), str(abi_vol), (float(pw_vol)-float(abi_vol))/float(pw_vol) ) )
#          if 'nan' in [abi_vol, abi_en, pw_vol, pw_en]:
#              pass
#          else:
#              pw_FE  = float( pw_en  ) - float( pw_1atom_en  )
#              abi_FE = float( abi_en ) - float( abi_1atom_en )
#              FE_LOG.write( '{}\t{:10s}\t{:10s}\t{}\n'.format(spgr, str(pw_FE), str(abi_FE), (float(pw_FE)-float(abi_FE))/float(pw_FE) ) )
#        except( KeyError ): 
#          print( spgr, ' missing' )  
    VOL_LOG.close()
    FE_LOG.close()
    column = []
    for line in open( '{}/Formation_energy.dat'.format( path2logs ) , 'r').readlines():
        if not line.startswith( '##' ): 
           column.append( line.split('\t') )
    try:
        column.sort(key=lambda x: x[1])
        print( 'lowest pw  formation energy: spgr = {}, E = {}'.format( column[0][0], column[0][1] ))
        column.sort(key=lambda x: x[2])
        print( 'lowest abi formation energy: spgr = {}, E = {}'.format( column[0][0], column[0][2] ))
    except(IndexError):
        pass

def abi_errors():
    return  { 'symmetry operation does not preserve geometry'     : 0 ,
              'overflow'                                          : 0 ,
              'symmetry error'                                    : 0 ,
              'reduce strprecon'                                  : 0 ,
              'try larger ngfft or smaller ecut'                  : 0 ,
              'modify rprim, acell and/or symrel'                 : 0 ,
              'check the folder'                                  : 0 ,
              'abi_xpotrf, info=1'                                : 0 ,
              'abi_xheev, info= 2'                                : 0 ,
              'larger lattice vectors/dilatmx'                    : 0 ,
              'It was not possible to find Fermi energy'          : 0 ,
              'unknown'                                           : 0 }

def pw_errors():
    return { 'two or more atoms overlap'                          : 0 ,
             'group not recognized'                               : 0 ,
             'Input ibrav not compatible with space group number' : 0 ,
             'Wrong classes for D_4h'                             : 0 ,
             'atoms differ by lattice vector'                     : 0 ,
             'wrong celldm'                                       : 0 ,
             'wrong number of columns in ATOMIC_POSITIONS'        : 0 ,
             'Not enough space allocated for radial FFT'          : 0 ,
             'not orthogonal operation'                           : 0 ,
             'symmetry not preserved'                             : 0 ,
             'dE0s is positive'                                   : 0 ,
             'smooth g-vectors missing !'                         : 0 ,
             'S matrix not positive definite'                     : 0 ,
             'charge is wrong'                                    : 0 ,
             'internal error'                                     : 0 ,
             'too many bands are not converged'                   : 0 ,
             'eigenvectors failed to converge'                    : 0 ,
             'problems computing cholesky'                        : 0 ,
             'unknown'                                            : 0 }

if __name__ == '__main__':
   main()
