# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 22:10:30 2022
*基于深度学习的超密集边缘计算网络(UDN)的卸载和资源调度*
* Single-Agent , for the task processings of user equipments in each Region
"""
import math
import random
import numpy as np

#****************************Env Parameter************************************#

length = 2000  # 通信范围长度（2000m）
width = 2000  # 通信范围宽度 (2000m)

h_micro = 10  # 微基站高度
h_macro = 10  # 宏基站高度
channel=4
B = 2e7  # 单信道带宽固定20MHz  1MHz = 10^6Hz


M = 4   # 单信道最大复用用户数量
S = 6   # BS数量  小小区数量  4-8
N = 2*S  # UE数量  每个小小区2个用户  8-16

f_ue = 10e8    # 单位HZ UE的最大CPU计算频率1-3 GHz      T_loc=CPUs of task/f_ue
f_mec = 600e8  # 单位HZ  MEC的最大CPU计算频率60-80 GHz  
f_mc = 1200e8  # 单位HZ  宏基站最大CPU计算频率120GHz

ncycle_per_bit = 1000  # 1 bit计算需要的CPU cycles数量 1000-2000  (CPUs of task=bits of task * cycle_per_bit)
κ = 10 ** (-27)     # 芯片结构对cpu处理的影响因子κ   E_loc=κ*f_ue*CPUs of task
E_max = 1


p_uplink_ue = 4     # UE上行发射最大功率 4-6W
p_uplink_bs = 20      # BS发射功率为20-25W
p_noisy_nlos = 10 ** (-13)  # 白噪声功率-100dBm
g_unit = 1e-5  # 距离为1m时的信道增益g:-50dB = 1e-5     g_s=g_unit/distance**2


T = 400  # 周期400s   T {1, 2,...,400}
slots = 10  # 每个时隙10s
slot_num = int(T / slots)  # 400个时隙


delay_factor = 0.7  # 计算时间延迟权重
energy_factor = 1-delay_factor  # 能耗权重


bs_pos=np.random.randint(0, width, size=[S, 2]) #微基站坐标
macro_bs = [1000, 1500]



a_dim = (N, 2)  # 动作向量维度 N*2维,每一行代表一个UE 每个UE动作两部分：卸载决策（本地0和边缘计算1）、发射功率选择）
s_dim = (N, 3)   # 状态向量维度 N*3 维，每一行代表一个用户UE,包含三个状态信息，任务大小,任务最大延迟, 前一次选择动作反馈的上行速率)

#*****************************************************************************#

class UDNEnv():
    def __init__(self):
        """
        初始化UDN环境
        包括BS/UE 数量位置,系统带宽和信道数量，智能体状态和动作空间维数,初始状态
        系统收集计算任务大小，可用信道以及每个信道复用用户情况,BS mec服务器剩余计算资源
        """
        self.num_ue = N #总用户集
        self.num_bs = S #基站集
        self.a_dim = a_dim #动作
        self.s_dim = s_dim  #状态
        self.t_factor = 0.7  # 计算时间延迟权重
        self.e_factor = 0.3  # 能耗权重
        self.total_cost=0  #时延和能耗总成本
        self.time_delay=[] #时延
        self.data_rate_uplink = np.zeros((1, self.num_ue)).astype('float64')
        self.f_ue = f_ue  #UE CPU主频
        self.f_mec = f_mec   #MEC服务器主频（最大）
        self.f_mc = f_mc  #宏基站CPU主频
        self.e_max=E_max  #最大能耗
        self.bs_pos=bs_pos  #小基站坐标
        self.p_uplink_ue=p_uplink_ue #UE最大发射功率
        self.p_uplink_bs=p_uplink_bs  #小基站发射功率（固定）
        self.T=T  # 总时隙
        self.slots=slots  #单个时隙长度
        self.slot_num=slot_num  #时隙数量
        self.reward_list=np.zeros((1,self.num_ue)).astype('float64')  #奖励列表
        self.total_reward=0  # 奖励统计
        self.total_td=0  #时延统计
        self.total_e=0   #能耗统计
       

        
    def reset(self):
        # 更新系统状态:所有UE任务大小task_size、最大容忍延迟time_delay、前一次状态UE上行速率data_rate_link
        self.reset_env()
        task_list = []
        self.task=[]
        task_list += [self.ue_task[i].tolist() for i in range(self.num_bs)]
        for i in range(len(task_list)):
             self.task+=task_list[i]
        self.state = np.array(self.task,dtype='float64').reshape(1,self.num_ue)
        self.state = np.append(self.state, self.time_delay, axis=0)
        self.state = np.append(self.state, self.data_rate_uplink, axis=0)
        self.state=self.state.T
        # self.state_normal()
        return self.state   #(task_size,time_delay,data_rate) N*3    
    
    # def state_normal(self):  
    #     # State Normalization
    #     self.state[:,0]= (self.state[:,0]-np.min(self.state[:,0]))/(np.max(self.state[:,0])-np.min(self.state[:,0]))
    #     if 0 not in self.state[:,2]:
    #         self.state[:,2]= (self.state[:,2]-np.min(self.state[:,2]))/(np.max(self.state[:,2])-np.min(self.state[:,2]))
   
    def reset_env(self):

      # 重置UE和BS通信环境,随机生成每个小区UE的坐标和对应大小的计算任务及完成最大延迟\
        # # seq_a = np.random.randint(2, 6)
        # # seq_b = np.random.randint(seq_a+2, 8)
        # seq_a=5   #N1:5  N2:4  N3:3
        # seq_b=9
        # self.ue_id = {0: list(range(0, seq_a)), 1: list(range(seq_a, seq_b)), 2: list(range(seq_b, N))}
        # self.ue_pos = {0: loc[:seq_a:], 1: loc[seq_a:seq_b,:], 2: loc[seq_b:,:]}  # UE的坐标（x,y)
        self.ue_id={i:list(range(i*2,(i+1)*2)) for i in range(self.num_bs)} #每个小小区2个用户
        loc = np.random.randint(0, width, size=[self.num_ue, 2])  # UE的位置不固定,每次重置重新生成
        self.ue_pos = {i: loc[i*2:(i+1)*2] for i in range(self.num_bs)}  # UE的坐标（x,y)
        
        ue_bs_distance=[np.sum(np.power(self.ue_pos[num]-bs_pos[num],2),axis=1) for num in range(self.num_bs)]
       
        while False in [len(set(data))==len(data) for data in ue_bs_distance]: #判断是否存在距离相同的UE,如有重新生成坐标
            # np.random.seed(1234)
            loc = np.random.randint(0, width, size=[self.num_ue, 2]) 
            self.ue_pos = {i: loc[i*2:(i+1)*2] for i in range(self.num_bs)}  # UE的坐标（x,y)
            ue_bs_distance=[np.sum(np.power(self.ue_pos[num]-bs_pos[num],2),axis=1) for num in range(self.num_bs)] 
    
        task = np.random.randint(3000000,4000000,self.num_ue)  #随机生成计算任务大小 600-1000 KB 
        # self.ue_task = {0: task[:seq_a], 1: task[seq_a:
        #     seq_b], 2: task[seq_b:]}  # 每个小小区的UE计算任务
        
        self.ue_task={i:task[i*2:(i+1)*2] for i in range(self.num_bs)}
        self.time_delay = np.random.uniform(0.5,0.75,(1,self.num_ue)) #随机为UE指定任务完成最大延迟0.5-0.75s间
        # self.time_delay=np.full((1,self.num_ue),fill_value=0.6) #对所有UE完成计算任务固定最大时延均为0.5-0.75s
        self.task_size = [len(self.ue_task[i]) for i in range(self.num_bs)] #任务大小

    def get_data_rate_ue(self, ue_id, bs_num,offload, uplink_ratio, offload_task):

        # 计算UE和BS之间的上行传输速率
        index = self.ue_id[bs_num].index(ue_id)

        bs_loc = self.bs_pos[bs_num]  # 当前UE关联的BS的坐标
        distance = (self.ue_pos[bs_num][index][0]-bs_loc[0])**2 + \
                          (self.ue_pos[bs_num][index][1]-bs_loc[1])**2+h_micro**2  # 计算距离
            
        channel_ue_id=[self.task.index(tasks) for tasks in offload_task]
        channel_ue_index=[self.ue_id[bs_num].index(ue) for ue in channel_ue_id if  offload[ue]==1] 
        
        distance_ue = [(self.ue_pos[bs_num][index][0]-bs_loc[0])**2 + (self.ue_pos[bs_num][index][1] -
                             bs_loc[1])**2+h_micro**2 for index in  channel_ue_index]
          
        distance_sorted=sorted(distance_ue) #信道增益按照降序排列，按照增益顺序受到干扰，最后解调信号无干扰
       
        interrupt_ue_index=[distance_ue.index(dist) for dist in distance_ue if dist>distance]#获取受到干扰的同一小区其他UE的ID
        interrupt_ue_id=[self.ue_id[bs_num][i] for i in interrupt_ue_index]
        
    
        if len(interrupt_ue_id)==0: #无干扰
            distance_others=0
            interrupt=0
            p_up_link_ue= uplink_ratio[ue_id]*self.p_uplink_ue
            gamma_ue=p_up_link_ue * abs(g_unit/distance) / (p_noisy_nlos) 
        else:
            
            distance_others = [(self.ue_pos[bs_num][index][0]-bs_loc[0])**2 + (self.ue_pos[bs_num][index][1] -
                                 bs_loc[1])**2+h_micro**2 for index in  interrupt_ue_index]
            p_up_link = [p_uplink_ue*uplink_ratio[ue] for ue in  interrupt_ue_id  if  offload[ue]==1]
    
            interrupt = np.sum(np.array(p_up_link) * g_unit/np.array(distance_others))  # 计算小区用户信道干扰
            p_up_link_ue= uplink_ratio[ue_id]*self.p_uplink_ue
            gamma_ue = p_up_link_ue * abs(g_unit/distance) / (interrupt + p_noisy_nlos)  # UE和BS之间的SINR信干噪比
              
        return B*math.log2(1+gamma_ue)  # 返回UE在当前信道的上行速率
        
    def get_data_rate_bs(self, bs_num, bs_adj_num, p_uplink_bs=3):
         # 计算BS之间的传输速率

        distance = (bs_pos[bs_num][0]-bs_pos[bs_adj_num][0])**2 + (bs_pos[bs_num][1]-bs_pos[bs_adj_num][1])**2
        # 微基站之间SINR(采用NOMA技术忽略小区之间干扰)
        gamma_bs = p_uplink_bs*abs(g_unit/distance)/p_noisy_nlos
        return B*math.log2(1+gamma_bs)

    def get_data_rate_mc(self, bs_num, p_uplink_bs=3):
        # 计算BS和Macro-BS之间的传输速率
        distance = (bs_pos[bs_num][0]-macro_bs[0])**2 + (bs_pos[bs_num][1]-macro_bs[1])**2+(h_macro-h_micro)**2
        # 微基站-宏基站SINR(NOMA技术忽略小区之间干扰)
        gamma_mc = p_uplink_bs*abs(g_unit/distance)/p_noisy_nlos
        return B*math.log2(1+gamma_mc)

    def get_action_ue(self,action):
         
         offload=[1 if action[i,0]>=0.5 else 0  for i in range(action.shape[0])]
         p_uplink_ratio=action[:,1]
         return offload,p_uplink_ratio  
        # # 每轮UE的动作集合
        # offload_num =np.random.randint(0, 2,(self.num_ue,1))  # 任务卸载（本地0,MEC卸载1）
        # p_uplink_ratio =np.random.uniform(0.5, 1,(self.num_ue,1)) # 发射功率比例选择（0和1之间）
        # action=(offload_num,p_uplink_ratio)
        # return action  # 返回动作集合

    def queue(self,bs_num,ue_id,task_offload):  #对task按照增益从高到低进行优先级排序，假设基站先接收增益大的信号进行处理
        
        bs_loc = self.bs_pos[bs_num] 
        channel_ue_id=[self.task.index(tasks) for tasks in task_offload[bs_num]]
        channel_ue_index=[self.ue_id[bs_num].index(ue) for ue in channel_ue_id] 
        distance_ue = [(self.ue_pos[bs_num][index][0]-bs_loc[0])**2 + (self.ue_pos[bs_num][index][1] -
                             bs_loc[1])**2 for index in  channel_ue_index]

        distance_sorted=sorted(distance_ue)
        
        process_order=[channel_ue_id[distance_ue.index(dist)] for dist in distance_sorted]
        task_size_order=[self.task[i] for i in process_order]
        return task_size_order,process_order


    def step(self,action):

        remain_resource = {i: self.f_mec *ncycle_per_bit for i in range(self.num_bs)}  # 记录每个基站MEC剩余计算资源
        done=True
        data_rate_uplink = []
        cost=[]
        total_delay=[]
        e_cost=[]
        task_offload={bs_num:[] for bs_num in range(self.num_bs)} #每个小区卸载到基站处理的任务
        offload,uplink_ratio=self.get_action_ue(action)   #卸载决策和发射功率选择
        # multiplex={i:None for i in range(self.num_bs)}
        # seq=0
        # for bs_num in self.ue_id.keys():
        #     multiplex[bs_num]=offload[seq:seq+(self.ue_id[bs_num])].count(1)
        #     seq=len(self.ue_id[bs_num])
        # multiplex_channel={i if multiplex[i]<=M else M for i in range(self.num_bs)} # 保证最大复用数为M
        reward_list=[]
        
        for bs_num in range(self.num_bs):
           task_offload[bs_num]=[self.task[ue] for ue in self.ue_id[bs_num] if offload[ue]==1]
           if len(task_offload[bs_num])>M:
               task_offload[bs_num]=task_offload[bs_num][:M]
        
        for bs_num in range(self.num_bs):
           for ue_id in self.ue_id[bs_num]:
               mec_ts = 0  #微基站资源转移状态
               mc_ts = 0   #宏基站转移状态
               reward=0
               t_local,t_mec,t_mec_ts, t_mc=(0,0,0,0)
               e_local,e_mec,e_mec_ts,e_mc=(0,0,0,0)
               data_rate_uplink_ue=0
               task_size = self.state[ue_id,0]  # 获取该用户的计算任务
               offload_num=offload[ue_id]   # 为UE随机生成卸载策略
               p_uplink_ratio=uplink_ratio[ue_id]#为UE随机生成发射功率
               
               if task_size in task_offload[bs_num]:
                     total_task_size=sum(task_offload[bs_num])  #总计算任务大小
                     task_size_order,process_order=self.queue(bs_num,ue_id,task_offload)
                     cum_size_ue=sum(task_size_order[:process_order.index(ue_id)+1])  #当前UE和之前UE总共处理任务大小
               
               if offload_num==0 or task_size not in task_offload[bs_num]: #超过信道最大复用数
                       # 本地计算耗能和时延 f_ue:UE的计算频率 ：单位bit处理所需cpu圈数 ncycle_per_bit
                       t_local = (1-offload_num)*task_size / (self.f_ue/ncycle_per_bit)  # CPU周期数处理时间计算
                       e_local = (1-offload_num)*κ*self.f_ue**2 * t_local  # 本地计算能耗  r:芯片结构对cpu处理的影响因子
                       
                  # 基站MEC传输耗能和时耗,不考虑基站计算能耗,只考虑传输耗能
                  # 当前基站资源充足情况
               if offload_num==1 and  cum_size_ue <= remain_resource[bs_num] and task_size in task_offload[bs_num]:
                       remain_resource[bs_num] = self.f_mec * ncycle_per_bit-task_size  # 资源被当前UE占用，更新MEC剩余资源
                       # t_mec_com=offload_num*task_size/(self.f_mec/ ncycle_per_bit) # 在UAV边缘服务器上计算时延
                       t_mec_com = sum(task_offload[bs_num]) *  ncycle_per_bit/self.f_mec  # 在MEC边缘服务器上分配资源计算时延
                       data_rate_uplink_ue = self.get_data_rate_ue(ue_id, bs_num,offload, uplink_ratio,task_offload[bs_num]) #计算上行吞吐量
                       t_mec_trans = offload_num*task_size/data_rate_uplink_ue #传输延迟
                       e_mec_trans = offload_num*task_size*p_uplink_ue*p_uplink_ratio/data_rate_uplink_ue  #传输能耗
                       mec_ts = 0  #不需要offload到其他微基站
                       mc_ts = 0    # 不需要offload到宏基站
                       
                       t_mec = (t_mec_trans+t_mec_com)*(1-mec_ts)*(1-mc_ts)  #总时延
                       e_mec = e_mec_trans*(1-mec_ts)*(1-mc_ts)    #总能耗
                 
                    # 当前基站资源不足,任务迁移至有资源的基站
               if offload_num==1 and cum_size_ue>remain_resource[bs_num]: 
                       bs_adj = [num for num in range(
                           self.num_bs) if num != bs_num and task_size <= remain_resource[num]-sum(task_offload[num])] #其他微基站还有资源
                       if len(bs_adj) > 1:
                           # 附近微基站有计算资源，通过计算距离选择较近的基站进行任务传递
                             mec_ts = 1
                             distance_adj = [(bs_pos[bs_num][0]-bs_pos[bs_adj[i]][0])**2+(bs_pos[bs_num][1]-bs_pos[bs_adj[i]][1])**2 for i in range(len(bs_adj))]
                             bs_adj_num = bs_adj[distance_adj.index(min(distance_adj))]  # 获得距离较近的基站编号
                             data_rate_bs = self.get_data_rate_bs(bs_num, bs_adj_num, p_uplink_bs=3)
                             t_bs_trans = offload_num*task_size/data_rate_bs
                             t_bs_com = offload_num*task_size /(sum(task_offload[bs_adj_num])+task_size) * (self.f_mec / ncycle_per_bit)  # 当前基站bs正在处理的计算任务大小
                             data_rate_uplink_ue = self.get_data_rate_ue(ue_id, bs_num,offload, uplink_ratio,task_offload[bs_num])
                             t_mec_trans = offload_num*task_size/data_rate_uplink_ue
                            
                             data_rate_uplink_ue = self.get_data_rate_ue(ue_id, bs_num,offload, uplink_ratio,task_offload[bs_num])
                             e_mec_trans = offload_num*task_size*p_uplink_ue*p_uplink_ratio/data_rate_uplink_ue 
                            
                             t_mec_ts = (t_mec_trans+t_bs_trans+t_bs_com)*mec_ts
                             e_mec_ts = p_uplink_bs*offload_num*task_size*mec_ts/data_rate_bs+e_mec_trans
                      
                       if len(bs_adj) == 0:  
                           # 所有微基站计算资源耗尽,发送至宏基站处理
                             mc_ts = 1
                             data_rate_uplink_mc = self.get_data_rate_mc( bs_num, p_uplink_bs=3)
                             t_mc_trans = offload_num*task_size/data_rate_uplink_mc
                             t_mc_com = offload_num*task_size / (self.f_mc / ncycle_per_bit)
                             data_rate_uplink_ue = self.get_data_rate_ue(ue_id, bs_num,offload, uplink_ratio,task_offload[bs_num])
                             t_mec_trans = offload_num*task_size/data_rate_uplink_ue
                             data_rate_mc=self.get_data_rate_mc(bs_num, p_uplink_bs=3)
                             e_mec_trans = offload_num*task_size*p_uplink_ue*p_uplink_ratio/data_rate_uplink_ue 
                             
                             t_mc = (t_mc_trans+t_mc_com+t_mec_trans)*mc_ts
                             e_mc = p_uplink_bs*offload_num*task_size*mc_ts/data_rate_mc*mc_ts+e_mec_trans
               
                 
               
               if offload_num==1 and mec_ts==0 and mc_ts==0:
                   t_mec = (t_mec_trans+t_mec_com)*(1-mec_ts)*(1-mc_ts) 
                   e_mec = e_mec_trans*(1-mec_ts)*(1-mc_ts)
               
              # 计算最终能耗和时延成本总和
               total_td = t_local+t_mec+t_mec_ts+t_mc
               total_e = e_local+e_mec+e_mec_ts+e_mc
               total_cost=self.t_factor*total_td+self.e_factor* total_e
               
               if total_td>slots: #单个UE处理时间超过时隙
                  done=False
                  break
               
               
            # 计算最大延迟和最大能耗是否满足条件,设置奖励
               td_max=max(t_local,t_mec,t_mec_ts,t_mc)  #最大延迟
               e_max=max(e_local,e_mec,e_mec_ts,e_mc)   #最大能耗
               if td_max>self.time_delay[0][ue_id] and e_max<self.e_max:# 高延迟低功耗
                       reward-=0
               if td_max<self.time_delay[0][ue_id] and e_max>self.e_max: #低延迟高功耗
                       reward-=0  
               if td_max>self.time_delay[0][ue_id]and e_max>self.e_max: #高延迟高功耗
                       reward-=0
               if td_max>=self.time_delay[0][ue_id]/2 and td_max<self.time_delay[0][ue_id] and  e_max<self.e_max: #低延迟低功耗
                      reward+=10
               if td_max<self.time_delay[0][ue_id]/2 and e_max<self.e_max:   #超低延迟低功耗
                      reward+=20
                      
            # 更新下一时刻状态
               reward_list.append(reward)
               data_rate_uplink.append(data_rate_uplink_ue)
               cost.append(total_cost)
               total_delay.append(total_td)
               e_cost.append(total_e)
        done=True        
        total_reward=sum(reward_list)
        total_delay=sum(total_delay)
        total_e=sum(e_cost)
        self.total_reward=total_reward
        self.reward_list=reward_list   
        self.data_rate_uplink=np.array(data_rate_uplink).reshape(1,self.num_ue)
        self.total_cost=sum(cost)
        self.total_td=total_delay
        self.total_e= total_e
        return total_reward,data_rate_uplink,done,total_delay,total_e

   
          
          

    
    
   
    
    
    
    
    
    
    