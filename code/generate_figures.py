#!/usr/bin/env python3
# import scipy.io
# mat = scipy.io.loadmat('file.mat')
# nuitka --recurse-on --python-version=3.6 daoct;
import sys
import os
import getopt
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.font_manager as font_manager
# import matplotlib
from matplotlib import rc
from matplotlib import rcParams


# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.sans-serif'] = 'cm'

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import tikzplotlib

import scipy.io as sio
import numpy as np
import numpy.matlib

import warnings
warnings.filterwarnings("ignore")
# config variables

nominal = sio.loadmat("../data/dmpc4rooms_ineq_chSetpoint__0_selfish__0_secure__0_.mat")
selfish = sio.loadmat("../data/dmpc4rooms_ineq_chSetpoint__0_selfish__1_secure__0_.mat")
corrected = sio.loadmat("../data/dmpc4rooms_ineq_chSetpoint__0_selfish__1_secure__1_.mat")

simK=nominal['simK'][0][0]

nominal_Wt=nominal['Wt']
nominal_xt=nominal['xt']
nominal_err=nominal['norm_err']
nominal_u=nominal['uHist'][0]


selfish_Wt=selfish['Wt']
selfish_xt=selfish['xt']
selfish_err=selfish['norm_err']
selfish_u=selfish['uHist'][0]


corrected_Wt=corrected['Wt']
corrected_xt=corrected['xt']
corrected_err=corrected['norm_err']
corrected_u=corrected['uHist'][0]

# ktotal=mat['ktotal'][0][0].astype(int)
nominal_J=nominal['J']
nominal_sumJ=np.sum(nominal_J,axis=0)

selfish_J=selfish['J']
selfish_sumJ=np.sum(selfish_J,axis=0)

corrected_J=corrected['J']
corrected_sumJ=np.sum(corrected_J,axis=0)

# print(nominal_sumJ)
nominalI=nominal_sumJ[0]
nominalII=nominal_sumJ[1]
nominalIII=nominal_sumJ[2]
nominalIV=nominal_sumJ[3]
nominal_global=np.sum(nominal_sumJ)

selfishI=selfish_sumJ[0]
selfishII=selfish_sumJ[1]
selfishIII=selfish_sumJ[2]
selfishIV=selfish_sumJ[3]
selfish_global=np.sum(selfish_sumJ)


correctedI=corrected_sumJ[0]
correctedII=corrected_sumJ[1]
correctedIII=corrected_sumJ[2]
correctedIV=corrected_sumJ[3]
corrected_global=np.sum(corrected_sumJ)

# Latex Total Cost Table
print("I &", round(nominalI),"& ",round(selfishI),"&",round(correctedI),"\\\\")
print("II &", round(nominalII),"& ",round(selfishII),"&",round(correctedII),"\\\\")
print("III &", round(nominalIII),"& ",round(selfishIII),"&",round(correctedIII),"\\\\")
print("IV &", round(nominalIV),"& ",round(selfishIV),"&",round(correctedIV),"\\\\")
print("Global &", round(nominal_global),"& ",round(selfish_global),"&",round(corrected_global),"\\\\")

# print("Nominal")
# print(np.sum((Wt[0:simK,0]-xt[0:simK,0])*(Wt[0:simK,0]-xt[0:simK,0])+uHist[0,-1,:,0]*uHist[0,-1,:,0])) # J1
# print(np.sum((Wt[0:simK,1]-xt[0:simK,1])*(Wt[0:simK,1]-xt[0:simK,1])+uHist[0,-1,:,1]*uHist[0,-1,:,1])) # J2
# print(np.sum((Wt[0:simK,2]-xt[0:simK,2])*(Wt[0:simK,2]-xt[0:simK,2])+uHist[0,-1,:,2]*uHist[0,-1,:,2])) # J3
# print(np.sum((Wt[0:simK,3]-xt[0:simK,3])*(Wt[0:simK,3]-xt[0:simK,3])+uHist[0,-1,:,3]*uHist[0,-1,:,3])) # J4

# print(np.sum((Wt[0:simK,0]-xt[0:simK,0])*(Wt[0:simK,0]-xt[0:simK,0])+uHist[0,-1,:,0]*uHist[0,-1,:,0])
#       +np.sum((Wt[0:simK,1]-xt[0:simK,1])*(Wt[0:simK,1]-xt[0:simK,1])+uHist[0,-1,:,1]*uHist[0,-1,:,1]) # J2
#       +np.sum((Wt[0:simK,2]-xt[0:simK,2])*(Wt[0:simK,2]-xt[0:simK,2])+uHist[0,-1,:,2]*uHist[0,-1,:,2]) # J3
#       +np.sum((Wt[0:simK,3]-xt[0:simK,3])*(Wt[0:simK,3]-xt[0:simK,3])+uHist[0,-1,:,3]*uHist[0,-1,:,3])) # J4

# print("Selfish")
# print(np.sum((selfWt[0:simK,0]-selfxt[0:simK,0])*(selfWt[0:simK,0]-selfxt[0:simK,0])+selfuHist[0,-1,:,0]*selfuHist[0,-1,:,0])) # J1
# print(np.sum((selfWt[0:simK,1]-selfxt[0:simK,1])*(selfWt[0:simK,1]-selfxt[0:simK,1])+selfuHist[0,-1,:,1]*selfuHist[0,-1,:,1])) # J2
# print(np.sum((selfWt[0:simK,2]-selfxt[0:simK,2])*(selfWt[0:simK,2]-selfxt[0:simK,2])+selfuHist[0,-1,:,2]*selfuHist[0,-1,:,2])) # J3
# print(np.sum((selfWt[0:simK,3]-selfxt[0:simK,3])*(selfWt[0:simK,3]-selfxt[0:simK,3])+selfuHist[0,-1,:,3]*selfuHist[0,-1,:,3])) # J4

# print(np.sum((selfWt[0:simK,0]-selfxt[0:simK,0])*(selfWt[0:simK,0]-selfxt[0:simK,0])+selfuHist[0,-1,:,0]*selfuHist[0,-1,:,0]) # J1
#       +np.sum((selfWt[0:simK,1]-selfxt[0:simK,1])*(selfWt[0:simK,1]-selfxt[0:simK,1])+selfuHist[0,-1,:,1]*selfuHist[0,-1,:,1]) # J2
#       +np.sum((selfWt[0:simK,2]-selfxt[0:simK,2])*(selfWt[0:simK,2]-selfxt[0:simK,2])+selfuHist[0,-1,:,2]*selfuHist[0,-1,:,2]) # J3
#       +np.sum((selfWt[0:simK,3]-selfxt[0:simK,3])*(selfWt[0:simK,3]-selfxt[0:simK,3])+selfuHist[0,-1,:,3]*selfuHist[0,-1,:,3])) # J4

# print("Corrected")
# print(np.sum((correctWt[0:simK,0]-correctxt[0:simK,0])*(correctWt[0:simK,0]-correctxt[0:simK,0])+correctuHist[0,-1,:,0]*correctuHist[0,-1,:,0])) # J1
# print(np.sum((correctWt[0:simK,1]-correctxt[0:simK,1])*(correctWt[0:simK,1]-correctxt[0:simK,1])+correctuHist[0,-1,:,1]*correctuHist[0,-1,:,1])) # J2
# print(np.sum((correctWt[0:simK,2]-correctxt[0:simK,2])*(correctWt[0:simK,2]-correctxt[0:simK,2])+correctuHist[0,-1,:,2]*correctuHist[0,-1,:,2])) # J3
# print(np.sum((correctWt[0:simK,3]-correctxt[0:simK,3])*(correctWt[0:simK,3]-correctxt[0:simK,3])+correctuHist[0,-1,:,3]*correctuHist[0,-1,:,3])) # J4

# print(np.sum((correctWt[0:simK,0]-correctxt[0:simK,0])*(correctWt[0:simK,0]-correctxt[0:simK,0])+correctuHist[0,-1,:,0]*correctuHist[0,-1,:,0]) # J1
#       +np.sum((correctWt[0:simK,1]-correctxt[0:simK,1])*(correctWt[0:simK,1]-correctxt[0:simK,1])+correctuHist[0,-1,:,1]*correctuHist[0,-1,:,1]) # J2
#       +np.sum((correctWt[0:simK,2]-correctxt[0:simK,2])*(correctWt[0:simK,2]-correctxt[0:simK,2])+correctuHist[0,-1,:,2]*correctuHist[0,-1,:,2]) # J3
#       +np.sum((correctWt[0:simK,3]-correctxt[0:simK,3])*(correctWt[0:simK,3]-correctxt[0:simK,3])+correctuHist[0,-1,:,3]*correctuHist[0,-1,:,3])) # J4

# NOTE(accacio): Detection
# fig, axs = plt.subplots(2, 1)

# axs[0].plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0],simK+1,1),'-',drawstyle='steps-post')
# axs[0].plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,0],'-',drawstyle='steps-post')
# axs[0].plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,0],'-',drawstyle='steps-post')
# axs[0].scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,0],s=15,color='black')

# axs[0].legend(( '$w_{\mathrm{I}}(k)$','$y_{\mathrm{I}}^N(k)$','$y_{\mathrm{I}}^S(k)$','$y_{\mathrm{I}}^C(k)$'),loc='bottom center',ncol=4,fontsize=13)

# axs[0].set_xticks(np.arange(0,simK+1,2))
# axs[0].set_xlim([1, simK])
# axs[0].set_ylim([15, 27])
# axs[0].set_title('Air temperature in room I ($^oC$)',fontsize=16)

# axs[1].plot(np.arange(1,simK+1),1e-4*np.ones([simK,1]),'-',drawstyle='steps-post') # error line
# axs[1].plot(np.arange(1,simK+1),selfish_err[0:simK,0],'-',drawstyle='steps-post')
# axs[1].scatter(np.arange(1,simK+1),nominal_err[0:simK,0],color='magenta',s=10)
# axs[1].scatter(np.arange(1,simK+1),corrected_err[0:simK,0],color='black',s=10)

# axs[1].legend(( '$\epsilon_p$','$E_{\mathrm{I}}^N(k)$','$E_{\mathrm{I}}^S(k)$','$E_{\mathrm{I}}^C(k)$'),loc='center',ncol=4,fontsize=13)

# axs[1].set_title("Norm of error $\| \widehat{\\tilde{P}^{1}_I}[k]-\\bar{P}^{1}_{I}\|_{F}$",fontsize=16)

# axs[1].set_xticks(np.arange(0,simK+1,2))
# axs[1].set_xlim([1, simK])
# axs[1].set_xlabel('Time (k)',usetex=True,fontsize=16)

# fig.tight_layout()
# plt.savefig("../img/airtemp_roomI" + "/__ErrorWX_command_normErrH" +  ".pdf",bbox_inches='tight')
# plt.savefig("../img/airtemp_roomI" + "/__ErrorWX_command_normErrH" +  ".png",bbox_inches='tight')

# NOTE(accacio): control
fig, axs = plt.subplots(3, 1)
axs[0].plot(np.arange(1,simK+1),nominal_u,'-',drawstyle='steps-post') # error line
axs[0].set_xticks(np.arange(0,simK+1,2))
axs[0].set_xlim([1, simK])
axs[0].set_title('Applied control $u_i$ ($kW$) ',fontsize=16)
axs[0].text(6, 3, "Nominal", ha="center", va="center",  size=16,
    bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w", ec="gray", lw=2))

axs[0].legend(( 'I', 'II','III','IV'),loc='upper right',ncol=4,fontsize=13)
axs[1].plot(np.arange(1,simK+1),selfish_u,'-',drawstyle='steps-post') # error line
axs[1].set_xticks(np.arange(0,simK+1,2))
axs[1].set_xlim([1, simK])
axs[1].set_title('',fontsize=16)
axs[1].text(6, 3, "Selfish", ha="center", va="center",  size=16,
    bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w", ec="gray", lw=2))

axs[1].legend(( 'I', 'II','III','IV'),loc='upper right',ncol=4,fontsize=13)
axs[2].plot(np.arange(1,simK+1),corrected_u,'-',drawstyle='steps-post') # error line
axs[2].set_xticks(np.arange(0,simK+1,2))
axs[2].set_xlim([1, simK])
axs[2].set_title('',fontsize=16)
axs[2].text(8, 3, "+ Correction", ha="center", va="center",  size=16,
    bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w",ec="gray",lw=2))
axs[2].legend(( 'I', 'II','III','IV'),loc='upper right',ncol=4,fontsize=13)
axs[2].set_xlabel('Time (k)',usetex=True,fontsize=16)

fig.tight_layout()
plt.savefig("../img/airtemp_roomI" + "/control" +  ".pdf",bbox_inches='tight')
plt.savefig("../img/airtemp_roomI" + "/control" +  ".png",bbox_inches='tight')

# NOTE(accacio): Costs
# fig, axs = plt.subplots(4, 1)
# axs[0].plot(np.arange(1,simK+1),nominal_J,'-',drawstyle='steps-post') # error line
# axs[0].plot(np.arange(1,simK+1),np.sum(nominal_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[0].set_xticks(np.arange(0,simK+1,2))
# axs[0].set_xlim([1, simK])
# axs[0].set_title('',fontsize=16)
# axs[0].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=13)
# axs[1].plot(np.arange(1,simK+1),(selfish_J),'-',drawstyle='steps-post') # error line
# axs[1].plot(np.arange(1,simK+1),np.sum(selfish_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[1].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=13)
# axs[2].plot(np.arange(1,simK+1),(corrected_J),'-',drawstyle='steps-post') # error line
# axs[2].plot(np.arange(1,simK+1),np.sum(corrected_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[2].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=13)
# axs[3].plot(np.arange(1,simK+1),np.sum(nominal_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].plot(np.arange(1,simK+1),np.sum(selfish_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].plot(np.arange(1,simK+1),np.sum(corrected_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].legend(( 'N', 'S','C'),loc='bottom right',ncol=3,fontsize=13)

plt.show()
sys.exit()

# plt.savefig("../img/airtemp_roomI" + "/control" +  ".pdf",bbox_inches='tight')
# plt.savefig("../img/airtemp_roomI" + "/control" +  ".png",bbox_inches='tight')

# plot(reshape(subsystems.uHist(1,end,:,i),[simK 1]),linS{i},'Color',colors{i})

# axs[0].plot(np.arange(1,simK+1), Wt[0,0:0+simK],drawstyle='steps')
# axs[0].plot(np.arange(0,ktotal), W[0,0:0+ktotal],'r',drawstyle='steps')
# axs[0].plot(np.arange(2,ktotal+1), W[0,0:0+ktotal-1],'--r',drawstyle='steps')
# # axs[0].set_xlabel('Iteration (k)',usetex=True)
# axs[0].set_title('Temperature of room 1',fontsize=16)
# axs[0].set_ylim([18 ,22])
# axs[0].set_yticks(np.arange(18,23,1))
# axs[0].set_xticks(np.arange(0,ktotal+1,1))
# axs[0].set_xlabel('Time',usetex=True,fontsize=16)
# axs[1].plot(np.arange(1,ktotal+1), Y[1,0:0+ktotal],drawstyle='steps')
# axs[1].plot(np.arange(2,ktotal+1), W[1,0:0+ktotal-1],'--r',drawstyle='steps')
# axs[1].set_ylim([18 ,22])
# axs[1].set_yticks(np.arange(18,23,1))
# axs[1].set_xticks(np.arange(0,ktotal+1,1))
# # axs[1].set_xlabel('Iteration (k)',usetex=True)
# axs[1].set_title('Temperature of room 2',fontsize=16)
# # axs[2].plot(np.arange(0,ktotal+0), J[0,-1,:])
# # axs[2].set_title('Global cost $J^{\star}$')
# axs[1].set_xlabel('Time',usetex=True,fontsize=16)

# #
# #
# color="red"
# fig, axs = plt.subplots(2,1)
# axs[0].plot(np.arange(0,ktotal), Y[0,0:0+ktotal])
# # axs[0].set_xlabel('Iteration (k)',usetex=True)
# axs2 = axs[0].twinx()
# axs2.plot(np.arange(1,ktotal+1), W[0,0:0+ktotal],'--',drawstyle='steps',color=color)
# axs2.tick_params(axis='y', labelcolor=color)
# axs[0].set_title('Temperature of room 1',fontsize=16)

# axs2.set_ylim([19 ,22])
# axs[0].set_xlim([1 ,20])
# axs[0].set_ylim([16 ,18])
# axs[0].set_xticks(np.arange(1,20,1))
# axs[1].plot(np.arange(0,ktotal), Y[1,0:0+ktotal])
# # axs[1].set_xlabel('Iteration (k)',usetex=True)
# axs2 = axs[1].twinx()
# axs2.plot(np.arange(1,ktotal+1), W[1,0:0+ktotal],'--r',drawstyle='steps')
# axs2.tick_params(axis='y', labelcolor=color)
# axs[1].set_title('Temperature of room 2',fontsize=16)

# axs2.set_ylim([19 ,22])
# axs[1].set_ylim([17 ,18])
# axs[1].set_xlim([1 ,20])
# axs[1].set_xticks(np.arange(1,20,1))
# # axs[2].plot(np.arange(0,ktotal), J[0,-1,:])
# # axs[2].set_xlim([1 ,20])
# # axs[2].set_xticks(np.arange(1,20,1))
# # axs[2].set_ylim([60 ,110])
# # axs[2].set_title('Global cost $J^{\star}$',fontsize=16)
# axs[1].set_xlabel('Time',usetex=True,fontsize=16)
# fig.tight_layout()
# plt.savefig(outputFolder + "/" + os.path.basename(inputFile) + "__TempAndJtogether" +  ".pdf",bbox_inches='tight')
# plt.savefig(outputFolder + "/" + os.path.basename(inputFile) + "__TempAndJtogether" +  ".png",bbox_inches='tight')

# # EigenValues
# eigAest=mat['eigAestHist']

# plt.figure()
# plt.plot(np.arange(1,ktotal+1), eigAest[1:-1,0:0+ktotal].T,'*')

# # plt.ylim(1, 3)
# # plt.xlim(0, 20)

# plt.title('Estimated eigenvalues of $\\mathcal{A}$',usetex=True,fontsize=16)
# plt.xlabel('Time (k)',usetex=True,fontsize=16)
# # plt.legend(loc='right')

# plt.xticks(np.arange(1,21,1))



# plt.savefig(outputFolder + "/" + os.path.basename(inputFile) + "__eigAest" +  ".pdf",bbox_inches='tight')
# plt.savefig(outputFolder + "/" + os.path.basename(inputFile) + "__eigAest" +  ".png",bbox_inches='tight')

# # plt.show()
