import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def read_inputs():
    names=['mass1.txt','mass2.txt','penx.txt','peny.txt']
    # data_out=np.zeros((300,62,4))
    lines_out=[]
    for j in range(len(names)):
        with open('./mass_pendulumv2/'+names[j]) as f:
            lines = f.readlines()                
        lines_out.append(lines)
    data_out=[]

    for j in range(len(lines_out)):
        for i in range(len(lines_out[0])):
            str_list=lines_out[j][i].replace('\n','').split(',')
            float_list=[float(x) for x in str_list]
            data_out.append(float_list)
            # data_out[i,:,j]=float_list
    pos=[]
    vel=[]
    acc=[]
    force=[]
    index0=[650,20,0,77,0,550,0,241,445,0,321,0,30,60,55,840,50,50,46,30,700,33,37,51,33,1057,64,44,30,33]
    index1=[1450,1020,1000,950,900,1320,970,1120,1062,955,1260,950,1025,1020,980,1800,950,1100,950,1030,1650,1000,1040,950,970,2000,1000,1050,1020,960]
    data2=np.array(data_out)
    multiple=int(len(data2)/len(names))
    
    #  i in range(multiple):

    # for i in range(len(data_out[0])):
    #     pos.append(data2[index0[i]:index1[i],i])
    #     vel.append(data2[index0[i]+2000:index1[i]+2000,i])
        # plt.plot(np.array(data_out)[:2000,i])
        # plt.show()



    data=[]
    data_index=[0]
    count=0
    divisor=-250.
    for i in range(len(data_out[0])):
        m=[]
            # m.append(max(abs(data_out[:,i,h])))
        for j in range(multiple-1):
            # x=torch.tensor([data2[j,i]/-1.,data2[j+multiple,i]/-1.,(data2[j+2*multiple,i])/-1.,data2[j+3*multiple,i]/-1.],dtype=torch.float)
            # y=torch.tensor([data2[j+1,i]/-1.,data2[j+multiple+1,i]/-1.,(data2[j+2*multiple+1,i])/-1.,data2[j+3*multiple+1,i]/-1.],dtype=torch.float)                     
            x=torch.tensor([data2[j,i]/divisor,data2[j+multiple,i]/divisor,(data2[j+2*multiple,i])/divisor,data2[j+3*multiple,i]/-divisor],dtype=torch.float)
            y=torch.tensor([data2[j+1,i]/divisor,data2[j+multiple+1,i]/divisor,(data2[j+2*multiple+1,i])/divisor,data2[j+3*multiple+1,i]/-divisor],dtype=torch.float)            
            # x=torch.tensor([data2[j,i],data2[j+multiple,i],data2[j+2*multiple,i],data2[j+3*multiple,i]],dtype=torch.float)
            # y=torch.tensor([data2[j+1,i],data2[j+multiple+1,i],data2[j+2*multiple+1,i],data2[j+3*multiple+1,i]],dtype=torch.float)
            data.append([x,y])
            count+=1
        data_index.append(count)
        

    torch.save(data,os.path.join('./',f'data_exp_osc_02142023.pt'))
    # np.save('oscillation_ind.npy',data_index)
    # for j in range(64):
    #     for i in range(len(lines_out[0])-1):
    #         pos_out=lines_out[0].replace('\n','').split(',')

read_inputs()