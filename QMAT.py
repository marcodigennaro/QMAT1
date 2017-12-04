#!/scratch/snx3000/mdigenna/anaconda2/bin/python

#import sys
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

import os, shutil,math
import subprocess
import time
import re
import time
import numpy as np
import pandas as pd
import math
now   = time.strftime("%Y-%m-%d-%H:%M")

global elem
##scicore 
#script_dir    = '/scicore/home/lilienfeld/digennar/scripts'
#qmat_dir      = '/scicore/home/lilienfeld/digennar/Work_Space/QMAT'
#path2dict     = '/scicore/home/lilienfeld/digennar/DICT'
#daint 
script_dir    = '/users/mdigenna/scripts'
qmat_dir      = '/scratch/snx3000/mdigenna/QMAT'
path2dict     = '/users/mdigenna/DICT'
path2psp      = '/users/mdigenna/pspfiles/quantum_espresso/SSSP_acc_PBE'
##
path2logs     = os.path.join( qmat_dir , 'LOGS' )
calculation   = 'vc-relax'
templates_dir = os.path.join( qmat_dir, 'run_relax_Si_new' )
RUNS_dir      = os.path.join( qmat_dir, '{}_RUNS'.format(calculation) )

Ev2Ry = 0.073498618
Ry2Ev = 13.605698066  #1/0.073498618
Ha2Ev = 27.2114

tot_calc = 3000
import PW_class
from functions_QMAT import get_lowest_state, check_running_jobs, print_relative_error, abi_errors, pw_errors

def main():
    elem_list = [ 'B', 'Li', 'Be' , 'H', 'He' ]
    for elem in elem_list:
        ###############################################################
        ## Read Isolated atom results
        one_atom_pw          = PW_class.ISOLATED_SYS( elem, 'pw' )
        one_atom_pw_results  = one_atom_pw.read_isolated_pw_out()
        ##
        elem_dir = os.path.join( RUNS_dir,  elem )
        log_dir  = os.path.join( path2logs, calculation )
        for fold in [ path2logs, RUNS_dir, elem_dir, log_dir ]:
            if not os.path.exists( fold ):
                   os.makedirs( fold )
        lowest_energy_dict = [None, 0,0] 
        ##
        skip_list = check_running_jobs() 
        skip_list.sort()
        running_jobs = len( skip_list )
        columns      = [ 'spgr', 'status', 'Natoms', 'Tot. Vol.(Bo^3)', 'Vol/at', 'Tot. En.(eV)', 'Form. En.(eV)', 'Press.(GPa)', 'CPU time', 'PW error', 'Slurm error' ]
        list_of_idxs = os.listdir( templates_dir )
        list_of_idxs.sort()
        pd.options.display.float_format = '{:4.8f}'.format
        df = pd.DataFrame(columns=columns) #, index=list_of_idxs)
        fail_index = 0
        idx_index  = 0
        for idx in list_of_idxs: #135]:
            idx_df = pd.Series({'status': 'tamere'})
            tmp_label    = '{}_{}'.format(elem, idx)
            print( idx_index, tmp_label )
            if  tmp_label in skip_list:
                print( 'Running...' )
                idx_df = pd.Series({'status': 'Running'})
            else:
                source_dir  = os.path.join( templates_dir, idx )
                ##########################################################################
                ## SYSTEM CLASS
                system    = PW_class.PW( elem, idx )
                if os.path.exists( system.run_dir ):
                   log_lines = open( system.run_log,  'r' ).readlines() 
                   try:
                     spgr    = int( [line for line in open(system.template, 'r').readlines() if line.strip().startswith('space_group') ][0].split('=')[1] )
                   except(IndexError):
                     spgr    = float('nan')
                   Natoms, Volume, Vol_at, TotEn, Press, Form_En, CPU_time = float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
                   input_dict  = None
                   code_error, slurm_error, log_status, pw_status = None, None, None, None
                   done_search = re.search( 'JOB DONE.', log_lines[-2] )
                   check_num   = system.read_running_log()
                   ############################
                   ## SYSTEM is complete
                   if done_search:
                      log_status = 'Completed'
                      pw_status, results_dict = system.read_pw_out() 
                      Natoms   = results_dict['Natoms']
                      Volume   = results_dict['Tot. Vol.(Bo^3)']
                      TotEn    = results_dict['Tot. En.(eV)']
                      Press    = results_dict['Press.(GPa)']
                      CPU_time = results_dict['CPU time']
                      #Form_En = float('{:1.16e}'.format( TotEn - Natoms*one_atom_pw_results[0] ))
                      Form_En = float('{:1.16e}'.format( TotEn/Natoms - one_atom_pw_results[0] ))
                      if float(Form_En) < lowest_energy_dict[2]:
                         lowest_energy_dict = [idx, spgr, Form_En]
                      Vol_at  = Volume/Natoms
                      if pw_status in ['Not_Converged', 'Max_iter_acheived']:
                         with open( system.running_log, 'a' ) as RUN_LOG:
                            RUN_LOG.write( '{} {}\n'.format( check_num + 1, 8*'-' ) )
                            RUN_LOG.write( '{} {}\n'.format( 4*' ', pw_status ) )
                         if running_jobs  < tot_calc:
                            slurm_job_id = system.find_last_slurm()
                            system.modify_resubmit( pw_status, slurm_job_id )
                            running_jobs += 1
                      idx_df = pd.Series({'spgr':spgr, 'status': pw_status, 'Natoms' : Natoms , 'Tot. Vol.(Bo^3)' : Volume, 'Vol/at' : Vol_at, 'Tot. En.(eV)' : TotEn , 'Form. En.(eV)' : Form_En, 'Press.(GPa)' : Press, 'CPU time' : CPU_time })
                   ############################
                   ## SYSTEM not complete
                   else:
                      code_error   = system.read_code_error()
                      slurm_error  = system.read_slurm_error()
                      slurm_job_id = system.find_last_slurm()
                      with open( system.running_log, 'a' ) as RUN_LOG:
                           RUN_LOG.write( '{} {}\n'.format( check_num + 1, 8*'-' ) )
                           RUN_LOG.write( '{} {}\n'.format( 4*' ', slurm_job_id  ) )
                           RUN_LOG.write( '{} {}\n'.format( 4*' ', code_error    ) )
                           RUN_LOG.write( '{} {}\n'.format( 4*' ', slurm_error   ) )
                      fail_index += 1
                      print( '{}failed = {}, running = {}, check_run = {}, log status = {}'.format( 8*' ', fail_index, running_jobs, check_num, log_status ))
                      for error, label in zip( [ code_error, slurm_error], ['PW','Slurm'] ):
                          print( '{}{} error = {}'.format( 8*' ', label, error ) )
                      idx_df = pd.Series({ 'PW error' : code_error , 'Slurm error' : slurm_error })
                      if running_jobs  < tot_calc:
                             ##########################################################################
                             ## SYSTEM EXISTS
                             #if check_num < 20:
                                if   log_status == 'Empty' or log_status == 'Missing_Log':  ## empty log (this should never happen)
                                     system.update_running_log( 'resubmitting SAME file in {}\n'.format(system.run_dir))
                                     system.make_batch(elem)
                                     os.chdir( system.run_dir )
                                     subprocess.call( 'sbatch pw.sh', shell=True)
                                     running_jobs += 1
                                else:
                                     stop_search = re.search( 'stopping ...', log_lines[-1] )
                                     if stop_search:
                                        print( stop_search.group() )
                                        print( 'right here:', code_error )
                                        log_status = 'Stopped'
                                        if   code_error in ['not_orthogonal_operation', 'Unknown', 'wyckoff_position', 'symmetry_not_preserved', None, 'cholesky']:
                                             log_status = 'Skipping'
                                             pw_status  = code_error
                                             print('Skipping...')
#                                        elif code_error == 'too_many_r-vectors':
#                                             os.chdir( system.run_dir )
#                                             inp_lines = open( system.inp_file, 'r' ).readlines()
#                                             with open( 'pw.in', 'w+' ) as NI:
#                                                  for line in inp_lines:
#                                                      if line.startswith( 'A ' ):
#                                                         new_line = ' {}'.format( line )
#                                                      elif line.startswith( 'B ' ):
#                                                         new_line = ' {}'.format( line )
#                                                      elif line.startswith( 'C ' ):
#                                                         new_line = ' {}'.format( line )
#                                                      else:
#                                                         new_line = line
#                                                      NI.write( new_line )
                                        elif code_error in ['too_many_processors', 'S matrix not positive definite', 'inconsistent_number_of_sticks']:
                                             os.chdir( system.run_dir )
                                             batch_dict = system.read_batch_file() 
                                             batch_dict['ntasks-per-core'] = 1
                                             system.make_batch( elem, batch_dict )
                                             system.update_running_log( '    {}\n'.format(batch_dict))
                                             subprocess.call( 'sbatch pw.sh', shell=True)
                                             running_jobs += 1
                                             print('Resubmitting...')
                                        else:
                                             ####################################
                                             ## Jobs failed due to explicit error 
                                             ## will modify input file
                                             if code_error in ['too_many_r-vectors', 'too_many_bands_are_not_converged', 'Small_cell']:
                                                system.modify_resubmit( code_error, slurm_job_id )
                                                running_jobs += 1
                                                print('Resubmitting...')
#                                                print( merda_secca )
                                     else:
                                        log_status = 'Unfinished'
                                        ############################
                                        ## SYSTEM is uncomplete
                                        ############################
                                        ## Jobs failed due to Low resources
                                        ## will modify batch
                                        batch_dict  = system.read_batch_file()              ## read batch file
                                        print( batch_dict )
                                        if   slurm_error == 'Short_time':
                                             batch_dict['hours'] += 1
                                             system.make_batch( elem, batch_dict )
                                             os.chdir( system.run_dir )
                                             subprocess.call( 'sbatch pw.sh', shell=True)
                                             running_jobs += 1
                                             print('Resubmitting...')
                                        elif slurm_error == 'Low_memory':
                                             batch_dict['mem'] += 1
                                             system.make_batch( elem, batch_dict )
                                             os.chdir( system.run_dir )
                                             subprocess.call( 'sbatch pw.sh', shell=True)
                                             running_jobs += 1
                                             print('Resubmitting...')
                                        elif slurm_error in [ 'Unknown', 'Empty', 'Task_error', 'SIGCONT' ]:
                                             print( 'Warning, slurm error = ' , slurm_error, system.run_dir) 
                                             system.update_running_log( '    slurm error = {}\n'.format(slurm_error))
                                             print('Skipping...')
                                        else:
                                             print( 'Warning, slurm error = ' , slurm_error, system.run_dir) 
                                             print('Skipping...')
                                     #else:
                                     #    print( 'Warning, log status = {}'.format( log_status ) )
                      else:
                        log_status = 'Calculation Limit exceeded' #pass
                        print('Skipping...')
                else:
                   ##########################################################################
                   ## NEW SYSTEM
                   system.make_new(elem)
                   log_status = 'New' 
                   running_jobs += 1

            ##
            idx_index += 1
            df.loc[idx] = idx_df
            ##       

        df['Form. En.(eV)'] -= df['Form. En.(eV)'].min() 
        try:
          min_idx = df['Form. En.(eV)'].idxmin()
          min_idx_res = df.loc[df['Form. En.(eV)'].idxmin()]
        except(TypeError):
          min_idx, min_idx_res = float('nan'), float('nan')  

        os.chdir( qmat_dir ) 
        with open( os.path.join( log_dir, 'qmat_{}.log'.format( elem )),'w+') as outfile:
             outfile.write('## Isolated atom results:\n## Energy = {} (eV)\tPressure = {} (GPa)\n\n'.format(one_atom_pw_results[0], one_atom_pw_results[1]) )
             try:
               outfile.write('## Idx with lowest formation energy = {}, spgr = {} \n\n'.format(min_idx, min_idx_res['spgr']) )
             except(TypeError):
               pass
             df.to_string(outfile,columns=columns)
             outfile.write('\n')

       
if __name__ == '__main__':
    main()
