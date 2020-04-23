"""
	Inverse Design of Parameters from Schroedinger equation given some Eigenenergies.
"""
import numpy as np
import math
import argparse
# PyTorch utilities
import torch,sys
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('/home/jozef/Desktop/master_thesis/GitLab/cm-bilayerboundstates/HamiltonianModel/my_libs')
sys.path.append('/cluster/home/jbucko/master_thesis/my_libs')
from energy_lines import energy_minima
from waveft_class_optim import*

# own routines
import utils

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float
# reproducibility is good
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.RandomState(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


# main class
class ParametrizedHamiltonian(nn.Module):
	def __init__(self):
		super().__init__()

		##############----variables----#############
		self.s = 1
		self.m = 0
		self.tau = 1
		self.Rinnm = 20
		self.tinmeV = 400
		self.UinmeV = 60
		self.VinmeV = 50
		
		self.dimxi = 50
		self.dxi = 2/(self.dimxi+1)
		self.xi = torch.linspace(0.0, 3.0, self.dimxi, device=device, requires_grad=False)

		self.dimB = 10
		self.nE = 50

		self.BinTmin = 0.06
		self.BinTmax = 2.5
		self.dB = 2/(self.dimB+1)
		self.BinT = torch.linspace(self.BinTmin, self.BinTmax, self.dimB, device=device, requires_grad=False)
		self.ones = torch.ones((self.dimB,self.dimxi,4))
		###############################################


		##########---target energies----##########
		print('------------------------------------------------------\ntarget energies calculation...\n')
		t = time.time()
		target_energies_class = energy_minima(self.m,self.UinmeV,self.VinmeV,self.tau,self.s,self.Rinnm,self.tinmeV,self.BinTmin,self.BinTmax,self.dimB,self.nE)
		self.target_E = target_energies_class.calc_lines()[-1][-1]
		
		tt = time.time()
		print('\ntarget energies calculation finished after {:.4f} s...\n------------------------------------------------------\n'.format(tt-t))
		self.target_E = torch.tensor(self.target_E,device = device, dtype = dtype, requires_grad = False)
		##########################################



		##########----define parameters and hamiltonian derivatives----#########
		self.params = nn.Parameter(torch.tensor([40.0,40.0],device=device), requires_grad=True)
		self.HU = torch.eye(4,device = device, requires_grad = False,dtype = dtype)
		self.HV = self.tau/2*torch.diag(torch.tensor([1,1,-1,-1],device = device, requires_grad = False,dtype = dtype))
		#########################################################################



	def loss(self):
		gE = self.E-self.target_E
		return torch.matmul(gE,gE), gE*2*self.dB


	def adjoint_gradient(self):

		#########----energies and eigenstates for actual state----###########
		print('------------------------------------------------------\nparams energies and eigenstates calculation...\n')
		t = time.time()
		params_energies_class = energy_minima(self.m,self.params[0].detach().numpy(),self.params[1].detach().numpy(),self.tau,self.s,self.Rinnm,self.tinmeV,self.BinTmin,self.BinTmax,self.dimB,self.nE)
		self.E = params_energies_class.calc_lines()[-1][-1]

		params_wf_arr = []
		for i in range(self.dimB):
			params_wf = psi_complete(self.E[i],self.BinT[i].detach().numpy(),self.s,self.m,self.tau,self.Rinnm,self.tinmeV,self.params[0].detach().numpy(),self.params[1].detach().numpy(),1)
			d = params_wf.psi_sq_norm()
			#print('normalization computed...\n')
			params_wf_arr.append([params_wf.psisq_joint_elements(xi.detach().numpy())/d for xi in self.xi])

		self.params_wf_tensor = torch.tensor(params_wf_arr,dtype = dtype, requires_grad = False).squeeze(3)
		#print(self.params_wf_tensor.shape)
		tt = time.time()
		print('params energies and eigenstates calculation finished after {:.4f} s...\n------------------------------------------------------\n'.format(tt-t))
		self.E = torch.tensor(self.E,device = device, dtype = dtype, requires_grad = False)
		###################################################################################


		# evaluate loss
		loss, lossE = self.loss()


		"""
		only necessary to adjust this part if loss function contains states
		A = self.H - self.E*torch.eye(self.dim, device=device)
		torch.solve can be parellelized by batches https://pytorch.org/docs/stable/torch.html?highlight=torch.solve#torch.solve
		A. has the shape (self.da,self.dim,self.dim)
		Lambda, LU = torch.solve(b,A)
		Lambda = self.projection(Lambda.view(-1))
		"""

		#########----derivatives of hamiltonian and losses----##########
		EU = torch.einsum('ijk,kl,ijl->i', self.params_wf_tensor, self.HU, self.ones)
		EV = torch.einsum('ijk,kl,ijl->i', self.params_wf_tensor, self.HV, self.ones)
		#print(Ep.shape,lossE.shape)
		lossU = torch.matmul(lossE,EU)#.unsqueeze(0) # dL/dp = (del)L/(del)E * (del)E/(del)p
		lossV = torch.matmul(lossE,EV)#.unsqueeze(0) # dL/dp = (del)L/(del)E * (del)E/(del)p
		print('------------------------------------------------------\nloss: {}, lossU: {}, lossV: {}'.format(loss.data, lossU.data, lossV.data))
		################################################################
		return loss, lossU, lossV


if __name__ == '__main__':
	#torch.autograd.set_detect_anomaly(True)
	# parse terminal input
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--epochs', type=int, default=100,
		 help="How many iterations for the optimizer")
	args = parser.parse_args()


	# initialize the model
	hamil = ParametrizedHamiltonian()
	hamil.zero_grad()
	#utils.print_param(hamil)

	#optimizer = optim.Adam(hamil.parameters(), lr=1)
	optimizer = optim.SGD(hamil.parameters(), lr=1e-2, momentum=0.2)
	#optimizer = torch.optim.LBFGS(hamil.parameters(), lr=1, max_iter=50, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

	loss_save = torch.empty((args.epochs))
	paramU_save = torch.empty((args.epochs+1))
	paramV_save = torch.empty((args.epochs+1))
	for p in hamil.parameters():
		paramU_save[0] = p[0]
		paramV_save[0] = p[1]

	for i in range(args.epochs):

		print('######################################################\n##############---------epoch {}--------################\n######################################################\n'.format(i))
		loss, lossU, lossV = hamil.adjoint_gradient()

		# Reverse mode AD
		optimizer.zero_grad()
		# loss.backward()

		# Adjoint sensitivity method
		for p in hamil.parameters():
			print('parameters:',p.data)
			p.grad = torch.tensor([lossU.clone(),1*lossV.clone()],dtype = dtype)
			optimizer.step() # here we can put inside the loop as it loops once only
			print('after update:',p.data,'\n------------------------------------------------------\n')
			paramU_save[i+1] = p[0]
			paramV_save[i+1] = p[1]
		#optimizer.step()
		loss_save[i] = loss

		#print('parameters after the optimizer step:',hamil.parameters().data)

		# if i%(args.epochs/10) == 0:
		# 	print("# of epoch :{}, Loss = {}".format(i, loss))

	# utils.myplot(hamil)
	np.savetxt('losses_U_{}_V_{}.csv'.format(hamil.UinmeV,hamil.VinmeV),loss_save.detach().numpy())
	np.savetxt('paramU_U_{}_V_{}.csv'.format(hamil.UinmeV,hamil.VinmeV),paramU_save.detach().numpy())
	np.savetxt('paramV_U_{}_V_{}.csv'.format(hamil.UinmeV,hamil.VinmeV),paramV_save.detach().numpy())