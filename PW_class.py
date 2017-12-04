#!/scratch/snx3000/mdigenna/anaconda2/bin/python

import os
import shutil
import re
import subprocess
import time
import numpy as np
from itertools import islice
import pickle

from QMAT import templates_dir, RUNS_dir, script_dir, qmat_dir, path2dict, path2psp
from QMAT import Ha2Ev, Ry2Ev 
now   = time.strftime("%Y-%m-%d-%H:%M")

# covalent radii from Pyykko, P., & Atsumi, M. (2009). Molecular Single-Bond Covalent Radii for Elements 1-118. Chemistry - A European Journal, 15(1), 186-197. doi:10.1002/chem.200800987
# in Angstrom
z2Covalentradius = { 1:  0.32, 2:  0.46,  3: 1.33,  4: 1.02,  5: 0.85,  6: 0.75, 7:  0.71,
                     8:  0.63, 9:  0.64, 10: 0.67, 11: 1.55, 12: 1.39, 13: 1.26, 14: 1.16, 
                     15: 1.11, 16: 1.03, 17: 0.99, 18: 0.96, 19: 1.96, 20: 1.71, 21: 1.48, 
                     22: 1.36, 23: 1.34, 24: 1.22, 25: 1.19, 26: 1.16, 27: 1.11, 28: 1.1,   
                     29: 1.12, 30: 1.18, 31: 1.24, 32: 1.21, 33: 1.21, 34: 1.16, 35: 1.14, 
                     36: 1.17, 37: 2.1,  38: 1.85, 39: 1.63, 40: 1.54, 41: 1.47, 42: 1.38, 
                     43: 1.28, 44: 1.25, 45: 1.25, 46: 1.2,  47: 1.28, 48: 1.36, 49: 1.42,  
                     50: 1.4,  51: 1.4,  52: 1.36, 53: 1.33, 54: 1.31, 55: 2.32, 56: 1.96, 
                     57: 1.8,  58: 1.63, 59: 1.76, 60: 1.74, 62: 1.72, 63: 1.68, 64: 1.69, 
                     65: 1.68, 66: 1.67, 67: 1.66, 68: 1.65, 69: 1.64, 70: 1.7,  71: 1.62, 
                     72: 1.52, 73: 1.46, 74: 1.37, 75: 1.31, 76: 1.29, 77: 1.22, 78: 1.23, 
                     79: 1.24, 80: 1.33, 81: 1.44, 82: 1.44, 83: 1.51}

elem_dict = pickle.load( open( os.path.join( path2dict , "SSSP_acc_PBE.p" ) , "rb") )

class PW():
      def __init__( self, elem, idx ):
          self.elem = elem
          self.idx  = idx
          self.template_dir   = os.path.join( templates_dir,     self.idx )
          self.template       = os.path.join( self.template_dir, 'qe.in'  )
          self.run_dir        = os.path.join( RUNS_dir, elem,    self.idx )
          self.run_log        = os.path.join( self.run_dir,      'log_pw' )
          self.inp_file       = os.path.join( self.run_dir,      'pw.in'  )
          self.running_log    = os.path.join( self.run_dir,      'tmp.run_log' )
          self.check_num      = float('nan')
          self.pw_log_status  = 'initialized'
          self.pw_error       = 'initialized'
          self.batch_file     = os.path.join( self.run_dir,   'pw.sh' )
          self.batch_template = os.path.join( script_dir,     'pw.sh' )

      def make_new( self, elem ):
          print('new {} calculation: ({})'.format( self.elem, self.idx ) )
          os.makedirs( self.run_dir )
          Z, mass, psp = elem_dict[self.elem]
          radius_Si   = z2Covalentradius[14]
          radius_elem = z2Covalentradius[Z]
          template_lines = open( self.template, 'r' ).readlines()
          os.chdir(    self.run_dir )
          with open( 'pw.in', 'w+' ) as INP:
               for line in template_lines:
                   if   line.strip().startswith( 'pseudo_dir' ):
                        new_line = '  pseudo_dir = "{}"\n'.format(path2psp)
                   elif line.strip().startswith('wfcdir'):
                        new_line = '  !wfcdir = "./wf_out/"\n'
                   elif line.strip().startswith('nstep'):
                        new_line = '  nstep = 1000\n'
                   #elif line.strip().startswith('ecutrho'):
                   #     new_line = '  ecutrho = 300\n'
                   #elif line.strip().startswith('ecutwfc'):
                   #     new_line = '  ecutwfc = 30\n'
                   elif line.strip()[:2] == 'A ':
                        new_line = '  A = {}\n'.format( radius_elem/radius_Si * float(line.split('=')[1]))
                   elif line.strip()[:2] == 'B ':
                        new_line = '  B = {}\n'.format( radius_elem/radius_Si * float(line.split('=')[1]))
                   elif line.strip()[:2] == 'C ':
                        new_line = '  C = {}\n'.format( radius_elem/radius_Si * float(line.split('=')[1]))
#                   elif line.strip().startswith( 'ntyp '):
#                        new_line = '  ntyp = 1\n  nbnd = 100\n'
                   elif line.strip().startswith('Si  28.0855'):
                        new_line = '  {}  {}  {}\n'.format(elem, mass, psp)
                   elif line.strip().startswith('celldm(1)'):
                        new_line = '  celldm(1) = {}\n'.format( radius_elem/radius_Si * float(line.split('=')[1]))
                   elif line.strip().startswith('celldm(2)'):
                        new_line = '  celldm(2) = {}\n'.format( radius_elem/radius_Si * float(line.split('=')[1]))
                   elif line.strip().startswith('celldm(3)'):
                        new_line = '  celldm(3) = {}\n'.format( radius_elem/radius_Si * float(line.split('=')[1]))
                   elif line.strip().startswith('Si'):
                        new_line = line.replace('Si', elem)
#                   elif line.strip().startswith( 'etot_conv_thr' ):
#                        new_line = ''
#                   elif line.strip().startswith( 'forc_conv_thr' ):
#                        new_line = ''
#                   elif line.strip().startswith( 'cell_dynamics' ):
#                        new_line = ''
                   elif line.strip().startswith( 'conv_thr' ):
                        new_line = '{}  electron_maxstep = 1000\n'.format(line)
                   else:
                        new_line = line
                   INP.write( new_line ) 
          self.make_batch(elem)
          subprocess.call('sbatch pw.sh', shell=True)
          self.check_num = 0
          with open( self.running_log , 'w+' ) as LOG:
               LOG.write( '## {}_{}\n0 {}\nPW FILE created on {}\n'.format( self.elem, self.idx, 8*'-', now ) )
 
      def read_running_log( self ):
          if os.path.exists( self.run_dir ) :
             try:
                running_log_lines = open( self.running_log , 'r' ).readlines()
                check_num  = int([ int(line.split()[0]) for line in running_log_lines if line[0].isdigit() ][-1])
             except( FileNotFoundError ):
                   with open( self.running_log, 'w+' ) as LOG:
                        LOG.write( '## {}_{}\n0 {}\n LOG FILE created on {}\n'.format( self.elem, self.idx, 8*'-', now ) ) 
                   check_num = 1
          else:
             check_num = float('nan')
          return( check_num ) 

      def update_running_log( self, new_line ):
          if os.path.exists( self.run_dir ) :
             with open( self.running_log, 'a' ) as LOG:
                  LOG.write( '{}{}'.format( 4*' ', new_line.replace(qmat_dir, '.') ))

      def read_batch_file( self ):
          old_batch_lines = open( self.batch_file , 'r' ).readlines()
          old_batch_dict  = dict()
          for old_line in old_batch_lines:
              if   '--time'  in old_line:
                   old_time  = old_line.strip().replace('#SBATCH --time=','')
                   old_time_list = old_time.split(':')
                   old_batch_dict[ 'hours' ]    = int(old_time_list[0].strip().replace('"',''))
                   old_batch_dict[ 'minutes' ]  = int(old_time_list[1].strip().replace('"',''))
              elif '--nodes'   in old_line:
                   old_batch_dict[ 'nodes' ]  = int(old_line.split('=')[1].strip().replace('"',''))
#              elif '--ntasks'  in old_line:
#                   old_batch_dict[ 'ntasks' ] = int(old_line.split('=')[1].split()[0].strip().replace('"',''))
              elif '--qos'     in old_line:
                   old_batch_dict[ 'queue' ]  = str(old_line.split('=')[1].strip().replace('"',''))
              elif '--mem-per-cpu' in old_line:
                   old_batch_dict[ 'mem' ]    = int(old_line.split('=')[1].strip().replace('"','').replace('G',''))
          return( old_batch_dict )

      def make_batch( self, elem, param_dict = { 'queue' : '30min', 'hours' : 1, 'minutes': 30, 'mem': 1, 'nodes': 1, 'ntasks': 16 } ):
          batch_temp_lines = open( self.batch_template, 'r' ).readlines()
#          if 1 < int(param_dict['hours']) < 6: 
#             param_dict['queue'] = '6hours'
#          elif int(param_dict['hours']) > 6:
#             param_dict['queue'] = '1day'
          parameters = { '#SBATCH --job-name'    : '#SBATCH --job-name="{}_{}"  \n'.format(elem, self.idx ),
#                         '#SBATCH --qos'         : '#SBATCH --qos="{}"          \n'.format(param_dict['queue']),
                         '#SBATCH --time'        : '#SBATCH --time="{}:{}:00"   \n'.format(param_dict['hours'], param_dict['minutes']),
#                         '#SBATCH --mem-per-cpu' : '#SBATCH --mem-per-cpu="{}G" \n'.format(param_dict['mem']),
#                         '#SBATCH --ntasks'      : '#SBATCH --ntasks={}         \n'.format(param_dict['ntasks']),
                         '#SBATCH --nodes'       : '#SBATCH --nodes={}          \n'.format(param_dict['nodes'])}
          with open( os.path.join( self.run_dir, 'pw.sh') , 'w+' ) as ns:
                for line in batch_temp_lines:
                    if '=' in line:
                        if line.split('=')[0] in parameters.keys():
                           new_line = parameters[ line.split('=')[0] ]
                        else:
                           new_line = line
                    else:
                        new_line = line
                    ns.write( new_line )

      def read_input_file( self ): #, input_file ):
          input_list  = [ ' ibrav ', ' space_group ', ' A ', ' B ', ' C ',
                          ' nbnd ' , ' nat ', 
                          #'cosAB ', 'cosAB ', 'cosAC ', 'cosAC ', 'cosBC ', 'cosBC ',
                          ' celldm(1) ', ' celldm(2) ', ' celldm(3) ', ' celldm(4) ', ' celldm(5) ', ' celldm(6) ', 
                          ' ecutwfc '  , ' electron_maxstep ', ' nstep ' ]

          input_dict  = dict()
          input_lines = open( self.inp_file , 'r' ).readlines()
          for line in input_lines:
              for key in input_list:
                  match = re.search( '{}='.format(key), line )
                  if match:
                     input_dict[ key ] = float(line.split('=')[1])
          return( input_dict )

      def modify_resubmit( self, code_error, slurm_job_id ):

          os.chdir( self.run_dir )
          pw_resubmit_dict = { 'Small_cell'                       : { 'cell_factor' : 1.2 , 'delta_nbnd' :   0, 'delta_ecutwfc' :  0,  'delta_nstep' :   0, 'delta_electron_maxstep' :   0 },
                               'dE0s_is_positive'                 : { 'cell_factor' : 1.01, 'delta_nbnd' :   0, 'delta_ecutwfc' :  0,  'delta_nstep' :   0, 'delta_electron_maxstep' :   0 },
                               'S matrix not positive definite'   : { 'cell_factor' : 1.01, 'delta_nbnd' :   0, 'delta_ecutwfc' :  0,  'delta_nstep' :   0, 'delta_electron_maxstep' :   0 },
                               'too_many_r-vectors'               : { 'cell_factor' : 1.15, 'delta_nbnd' :   0, 'delta_ecutwfc' :  0,  'delta_nstep' :   0, 'delta_electron_maxstep' :   0 },
                               'too_many_bands'                   : { 'cell_factor' : 1.00, 'delta_nbnd' : -20, 'delta_ecutwfc' :  0,  'delta_nstep' :   0, 'delta_electron_maxstep' :   0 },
                               'Not_Converged'                    : { 'cell_factor' : 1.00, 'delta_nbnd' :   0, 'delta_ecutwfc' :  0,  'delta_nstep' :   0, 'delta_electron_maxstep' : 1000 },
                               'Max_iter_acheived'                : { 'cell_factor' : 1.00, 'delta_nbnd' :   0, 'delta_ecutwfc' :  0,  'delta_nstep' : 500, 'delta_electron_maxstep' :   0 },
                               'charge_is_wrong'                  : { 'cell_factor' : 1   , 'delta_nbnd' :  20, 'delta_ecutwfc' :  0,  'delta_nstep' :   0, 'delta_electron_maxstep' :   0 },
                               'too_many_bands_are_not_converged' : { 'cell_factor' : 1   , 'delta_nbnd' :   0, 'delta_ecutwfc' : 10,  'delta_nstep' :   0, 'delta_electron_maxstep' :   0 } }

          delta_dict = pw_resubmit_dict[ code_error ]
          input_dict = self.read_input_file()
          new_input_dict = {}
          for variable in [' A ', ' B ', ' C ', 'celldm(1) ', 'celldm(2) ', 'celldm(3) ']:
              try:
                new_input_dict[variable] = float(input_dict[variable]) * float(delta_dict['cell_factor'])
              except(KeyError):
                pass
          new_input_dict[' nstep '           ] = float(input_dict[' nstep '           ]) + int(delta_dict['delta_nstep'])
          new_input_dict[' ecutwfc '         ] = float(input_dict[' ecutwfc '         ]) + int(delta_dict['delta_ecutwfc'])
          new_input_dict[' electron_maxstep '] = float(input_dict[' electron_maxstep ']) + int(delta_dict['delta_electron_maxstep'])
          if new_input_dict == input_dict:
             print( same_dict )
          else:
             input_lines = open( self.inp_file, 'r' ).readlines()
             self.move_failed_job( code_error, slurm_job_id )
             with open( 'pw.in' , 'w+' ) as NEW_INP:
                  for line in input_lines:
                      key_word = ' {} '.format(line.split('=')[0].strip())
                      if   key_word in new_input_dict.keys():
                           new_line = '  {} = {}\n'.format( key_word.strip(), new_input_dict[ key_word ] )
                      elif line.strip().startswith( 'nat '):
                           if code_error == 'charge_is_wrong':
                              if ' nbnd ' in input_dict.keys():
                                new_input_dict[' nbnd '] = float(input_dict[' nbnd ']) + int(delta_dict['delta_nbnd'])
                              else:
                                Z, mass, psp = elem_dict[self.elem]
                                new_input_dict[' nbnd '] = float(input_dict[' nat ']) * int(Z) + 10
                              new_line   = '{}  nbnd = {}\n'.format(line, new_input_dict[' nbnd '] )
                           else:
                              new_line = line
                      else:
                          new_line = line
                      NEW_INP.write( new_line )
             self.update_running_log( 'code error = {}, modifying input file:\n    from {}\n    to   {}\n'.format( code_error, input_dict, new_input_dict ) )
             subprocess.call( 'sbatch pw.sh', shell=True)

      def find_last_slurm( self ):
          slurm_files  = [ tmp_file for tmp_file in os.listdir( self.run_dir ) if tmp_file.startswith( 'slurm' ) ]
          self.update_running_log( 'list of slurm files = {}\n'.format(slurm_files) )
          try:
             slurm_job_id = max([ elem.replace('slurm-','').replace('.out','') for elem in slurm_files ]) 
          except( ValueError ):
             slurm_job_id = float('nan') 
          if len(slurm_files) > 5:
             print( '{} Warning: {} slurm files in {}'.format( 12*' ', len(slurm_files), self.run_dir.replace(qmat_dir, '.') ) )
          return( slurm_job_id )

      def read_slurm_error( self ):
          slurm_msg = {         'DUE TO TIME LIMIT' : 'Short_time',
                        'Exceeded job memory limit' : 'Low_memory',
                        'unable to add task'        : 'Task_error',
                        'srun: got SIGCONT'         : 'SIGCONT'   }

          last_slurm   = self.find_last_slurm()
          #slurm_error = 'Missing'
          slurm_file   = os.path.join( self.run_dir, 'slurm-{}.out'.format(last_slurm) )
          slurm_error  = None
          slurm_lines  = open( slurm_file, 'r' ).readlines()
          if   os.stat(slurm_file).st_size == 0:
               slurm_error = 'Empty'
          else:
               for line in slurm_lines:
                   for key in slurm_msg.keys():
                       match = re.search( key, line )
                       if match:
                          slurm_error = slurm_msg[key]
          #if slurm_error:
          #   self.update_running_log( 'SLURM error = {}\n'.format(slurm_error) )
          return( slurm_error )

      def read_pw_out( self ):
          results_dict = dict()
          log_lines    = open( self.run_log, 'r' ).readlines()
          nat_cell_msg      = 'number of atoms/cell'
          converged_msg     = 'convergence achieved'
          non_converged_msg = 'convergence NOT achieved'
          max_iter_msg      = 'Maximum number of iterations reached, stopping'
          tot_energy_msg    = '!    total energy'
          volume_msg        = 'new unit-cell volume'
          cpu_time_msg      = 'PWSCF        :'
          press_msg         = 'total   stress'
          pw_status         = None #'lsdfaof'
          for line in log_lines:
              ## energy
              tot_energy_match = re.search( tot_energy_msg, line )
              if tot_energy_match:
                  results_dict['Tot. En.(eV)']   = float('{:1.16e}'.format(Ry2Ev*float(line.split()[4])))
              ## nat/cell
              nat_cell_match = re.search( nat_cell_msg, line )
              if nat_cell_match:
                 results_dict['Natoms'] = int(line.split('=')[1])
              ## volume
              volume_match = re.search( volume_msg, line )
              if volume_match:
                 results_dict['Tot. Vol.(Bo^3)'] = float(line.split()[4])
              ## max num iteration acheived
              max_iter_match = re.search( max_iter_msg, line )
              if max_iter_match:
                 pw_status = 'Max_iter_acheived'
              ## scf
              scf_match = re.search( converged_msg, line )
              if scf_match:
                 pw_status = 'Converged'
              non_scf_match = re.search( non_converged_msg, line )
              if non_scf_match:
                 pw_status = 'Not_Converged'
                 print( 'Warning: system NOT converged' )
              ## cpu time
              cpu_time_match = re.search( cpu_time_msg, line )
              if cpu_time_match:
                 cpu_time = line.split('CPU')[0].replace(cpu_time_msg, '').strip() 
                 cpu_time_dict = {}
                 h_pos = cpu_time.find('h')
                 m_pos = cpu_time.find('m')
                 s_pos = cpu_time.find('s')
                 if h_pos == -1:
                   cpu_time_dict[ 'h' ] = 0
                 else:
                   cpu_time_dict[ 'h' ] = float(cpu_time[:h_pos] )
                 cpu_time_dict[ 'm' ] = float(cpu_time[h_pos+1:m_pos] )
                 if s_pos == -1:
                   cpu_time_dict[ 's' ] = 0
                 else:
                   cpu_time_dict[ 's' ] = float(cpu_time[m_pos+1:s_pos] )
                 cpu_secs = 3600*float(cpu_time_dict['h']) + 60*float(cpu_time_dict['m']) + float(cpu_time_dict['s'])
                 results_dict['CPU time'] = cpu_secs 
                 #results_dict['CPU time'] = cpu_time_dict 
              ## Press
              press_match = re.search( press_msg, line )
              if press_match:
                 try:
                   results_dict['Press.(GPa)'] = 0.1*float(line.split('=')[1])
                 except(ValueError):
                   results_dict['Press.(GPa)'] = '********'
          #print( 'reading {}, pw_status = {}, dict = {}'.format( self.run_log, pw_status, results_dict ) )
          return( pw_status, results_dict )

      def read_code_error( self ):
          pw_err_dict = {  'overlap!'                                                  : 'two or more atoms overlap'                   ,
                           'group not recognized'                                      : 'group not recognized'                        ,
                           'Input ibrav not compatible'                                : 'Input ibrav not compatible'                  ,
                           'Wrong classes for D_4h'                                    : 'Wrong classes for D_4h'                      ,
                           'differ by lattice vector'                                  : 'atoms differ by lattice vector'              ,
                           'wrong celldm(4)'                                           : 'wrong celldm(4)'                             ,
                           'wrong number of columns in ATOMIC_POSITIONS'               : 'wrong number of columns in ATOMIC_POSITIONS' ,
                           'Not enough space allocated for radial FFT'                 : 'Small_cell'                                  ,
                           'not orthogonal operation'                                  : 'not_orthogonal_operation'                    ,
                           'some of the original symmetry operations not satisfied'    : 'symmetry_not_preserved'                      ,
                           'dE0s is positive which should never happen'                : 'dE0s_is_positive'                            ,
                           'smooth g-vectors missing !'                                : 'smooth g-vectors missing !'                  ,
                           'S matrix not positive definite'                            : 'S matrix not positive definite'              ,
                           'charge is wrong'                                           : 'charge_is_wrong'                             ,
                           'internal error'                                            : 'internal error'                              ,
                           'wyckoff position not found'                                : 'wyckoff_position'                            ,
                           'differ by lattice vector'                                  : 'differ by lattice vector'                    ,
                           'too many bands are not converged'                          : 'too_many_bands_are_not_converged'            ,
                           'eigenvectors failed to converge'                           : 'eigenvectors failed to converge'             ,
                           'too many bands, or too few plane waves'                    : 'too_many_bands'                              ,
                           'too many r-vectors'                                        : 'too_many_r-vectors'                          ,
                           'No plane waves found: running on too many processors?'     : 'too_many_processors'                         ,
                           'inconsistent number of sticks'                             : 'inconsistent_number_of_sticks'               ,
                           'problems computing cholesky'                               : 'cholesky'                                    } 

          err_dict   = pw_err_dict
          err_file   = os.path.join( self.run_dir, 'CRASH' )
          log_file   = os.path.join( self.run_dir, 'log_pw' )
          #stop_search = re.search( 'stopping ...', log_lines[-1] )
          code_error = None
          try:
             err_lines = [ line.strip() for line in open( err_file, 'r').readlines() ]
             for line in err_lines:
                 for key in pw_err_dict.keys():
                     match = re.search( key, line )
                     if match:
                        code_error = pw_err_dict[key]
          except(IOError):
             print( 'Warning: missing CRASH file' )
             log_lines   = [ line.strip() for line in open( log_file, 'r').readlines() ]
             for line in log_lines:
                 for key in pw_err_dict.keys():
                     match = re.search( key, line )
                     if match:
                        code_error = pw_err_dict[key]
             #code_error = 'Missing_Log'
          #if code_error:
          #   self.update_running_log( 'CODE error = "{}"\n'.format(code_error) )
          return( code_error )

      def move_failed_job( self, slurm_error, slurm_job_id ):
          os.chdir( self.run_dir  )
          slurm_file    = os.path.join( self.run_dir, 'slurm-{}.out'.format(slurm_job_id) )
          files_to_copy = ['pw.in', 'pw.sh', slurm_file, 'log_pw', 'CRASH' ]
          fail_dir      = os.path.join( self.run_dir, 'FAILED', slurm_error.replace(' ', '_').replace(',',''), str(slurm_job_id) )
          if os.path.exists( fail_dir  ):
              print( 'Warning, {} exists already'.format( fail_dir ) )
          else:
              os.makedirs( fail_dir  )
          for file_to_copy in files_to_copy:
              try:
                 shutil.copy( file_to_copy, fail_dir )
              except(IOError): #FileNotFoundError):
                 print( '{} WARNING {} not found'.format(12*' ', file_to_copy.replace(qmat_dir, './') ) )

#class ISOLATED_SYS(SYS):
#      def __init__( self, elem, spgr, instance, kpt_density, code ):
#          SYS.__init__( self, elem, spgr, instance, kpt_density, code )
class ISOLATED_SYS():
      def __init__( self, elem, code ):
          self.code        = code
          self.elem        = elem
          self.run_dir     = os.path.join( qmat_dir, 'Isolated_atom', self.code )
          self.run_log     = os.path.join( self.run_dir, 'log_{}'.format(self.code[0:3])) 
          self.log_lines   = open( self.run_log, 'r' ).readlines()
      
      def read_isolated_pw_out( self ):
          log_lines  = open( self.run_log, 'r' ).readlines()
          Tot_en_Ev  = float('{:1.16e}'.format(Ry2Ev*float( [line.strip() for line in log_lines if '!    total energy'         in line][-1].split()[4])))
          try:
            Press_GPa = 0.1*float([line.strip() for line in log_lines if 'total   stress  ' in line][-1].split()[5])
          except(ValueError):
            Press_GPa = '********'
          print( ' Tot energy = {} eV, Press = {} GPa'.format( Tot_en_Ev, Press_GPa ) )
          return[ Tot_en_Ev, Press_GPa ]

      def read_isolated_abi_out( self ):
          log_lines     = open( self.run_log, 'r' ).readlines()
          Vol_at, En_at, Press_GPa, scf_steps, cpu_time = ['nan']*5
          scf_steps     = 'need check'
          ##
          Tot_en_Ev        = Ha2Ev*float([line.strip() for line in log_lines if '  etotal' in line][0].split()[1])
          Press_GPa        = float([line.strip() for line in log_lines if '-Cartesian components' in line][0].split()[7])
          print( ' Tot energy = {} eV, Press = {} GPa'.format( Tot_en_Ev, Press_GPa ) )
          return[ Tot_en_Ev, Press_GPa ]
        
