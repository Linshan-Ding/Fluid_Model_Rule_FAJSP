from docplex.mp.model import Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import copy

"""测试实例数据"""
FT10C = [
    [0, 29, 1, 78, 2, 9, 3, 36, 4, 49, 5, 11, 6, 62, 7, 56, 8, 44, 9, 21],
    [0, 43, 2, 90, 4, 75, 9, 11, 3, 69, 1, 28, 6, 46, 5, 46, 7, 72, 8, 30],
    [1, 91, 0, 85, 3, 39, 2, 74, 8, 90, 5, 10, 7, 12, 6, 89, 9, 45, 4, 33],
    [1, 81, 2, 95, 0, 71, 4, 99, 6, 9, 8, 52, 7, 85, 3, 98, 9, 22, 5, 43],
    [2, 14, 0, 6, 1, 22, 5, 61, 3, 26, 4, 69, 8, 21, 7, 49, 9, 72, 6, 53],
    [2, 84, 1, 2, 5, 52, 3, 95, 8, 48, 9, 72, 0, 47, 6, 65, 4, 6, 7, 25],
    [1, 46, 0, 37, 3, 61, 2, 13, 6, 32, 5, 21, 9, 32, 8, 89, 7, 30, 4, 55],
    [2, 31, 0, 86, 1, 46, 5, 74, 4, 32, 6, 88, 8, 19, 9, 48, 7, 36, 3, 79],
    [0, 76, 1, 69, 3, 76, 5, 51, 2, 85, 9, 11, 6, 40, 7, 89, 4, 26, 8, 74],
    [1, 85, 0, 13, 2, 61, 6, 7, 8, 64, 9, 76, 5, 47, 3, 52, 4, 90, 7, 45]
]
La16C = [
    [1, 21, 6, 71, 9, 16, 8, 52, 7, 26, 2, 34, 0, 53, 4, 21, 3, 55, 5, 95],
    [4, 55, 2, 31, 5, 98, 9, 79, 0, 12, 7, 66, 1, 42, 8, 77, 6, 77, 3, 39],
    [3, 34, 2, 64, 8, 62, 1, 19, 4, 92, 9, 79, 7, 43, 6, 54, 0, 83, 5, 37],
    [1, 87, 3, 69, 2, 87, 7, 38, 8, 24, 9, 83, 6, 41, 0, 93, 5, 77, 4, 60],
    [2, 98, 0, 44, 5, 25, 6, 75, 7, 43, 1, 49, 4, 96, 9, 77, 3, 17, 8, 79],
    [2, 35, 3, 76, 5, 28, 9, 10, 4, 61, 6, 9, 0, 95, 8, 35, 1, 7, 7, 95],
    [3, 16, 2, 59, 0, 46, 1, 91, 9, 43, 8, 50, 6, 52, 5, 59, 4, 28, 7, 27],
    [1, 45, 0, 87, 3, 41, 4, 20, 6, 54, 9, 43, 8, 14, 5, 9, 2, 39, 7, 71],
    [4, 33, 2, 37, 8, 66, 5, 33, 3, 26, 7, 8, 1, 28, 6, 89, 9, 42, 0, 78],
    [8, 69, 9, 81, 2, 94, 4, 96, 3, 27, 0, 69, 7, 45, 6, 78, 1, 74, 5, 84]
]
Orb1C = [
    [0, 72, 1, 64, 2, 55, 3, 31, 4, 53, 5, 95, 6, 11, 7, 52, 8, 6, 9, 84],
    [0, 61, 3, 27, 4, 88, 2, 78, 1, 49, 5, 83, 8, 91, 6, 74, 7, 29, 9, 87],
    [0, 86, 3, 32, 1, 35, 2, 37, 5, 18, 4, 48, 6, 91, 7, 52, 9, 60, 8, 30],
    [0, 8, 1, 82, 4, 27, 3, 99, 6, 74, 5, 9, 2, 33, 9, 20, 7, 59, 8, 98],
    [1, 50, 0, 94, 5, 43, 3, 62, 4, 55, 7, 48, 2, 5, 8, 36, 9, 47, 6, 36],
    [0, 53, 6, 30, 2, 7, 3, 12, 1, 68, 8, 87, 4, 28, 9, 70, 7, 45, 5, 7],
    [2, 29, 3, 96, 0, 99, 1, 14, 4, 34, 7, 14, 5, 7, 6, 76, 8, 57, 9, 76],
    [2, 90, 0, 19, 3, 87, 4, 51, 1, 84, 5, 45, 9, 84, 6, 58, 7, 81, 8, 96],
    [2, 97, 1, 99, 4, 93, 0, 38, 7, 13, 5, 96, 3, 40, 9, 64, 6, 32, 8, 45],
    [2, 44, 0, 60, 8, 29, 3, 5, 6, 74, 1, 85, 4, 34, 7, 95, 9, 51, 5, 47]
]
Orb2C = [
    [0, 72, 1, 54, 2, 33, 3, 86, 4, 75, 5, 16, 6, 96, 7, 7, 8, 99, 9, 76],
    [0, 16, 3, 88, 4, 48, 8, 52, 9, 60, 6, 29, 7, 18, 5, 89, 2, 80, 1, 76],
    [0, 47, 7, 11, 3, 14, 2, 56, 6, 16, 4, 83, 1, 10, 5, 61, 8, 24, 9, 58],
    [0, 49, 1, 31, 3, 17, 8, 50, 5, 63, 2, 35, 4, 65, 7, 23, 6, 50, 9, 29],
    [0, 55, 6, 6, 1, 28, 3, 96, 5, 86, 2, 99, 9, 14, 7, 70, 8, 64, 4, 24],
    [4, 46, 0, 23, 6, 70, 8, 19, 2, 54, 3, 22, 9, 85, 7, 87, 5, 79, 1, 93],
    [4, 76, 3, 60, 0, 76, 9, 98, 2, 76, 1, 50, 8, 86, 7, 14, 6, 27, 5, 57],
    [4, 93, 6, 27, 9, 57, 3, 87, 8, 86, 2, 54, 7, 24, 5, 49, 0, 20, 1, 47],
    [2, 28, 6, 11, 8, 78, 7, 85, 4, 63, 9, 81, 3, 10, 1, 9, 5, 46, 0, 32],
    [2, 22, 9, 76, 5, 89, 8, 13, 6, 88, 3, 10, 7, 75, 4, 98, 1, 78, 0, 17]
]
Orb3C = [
    [0, 96, 1, 69, 2, 25, 3, 5, 4, 55, 5, 15, 6, 88, 7, 11, 8, 17, 9, 82],
    [0, 11, 1, 48, 2, 67, 3, 38, 4, 18, 7, 24, 6, 62, 5, 92, 9, 96, 8, 81],
    [2, 67, 1, 63, 0, 93, 4, 85, 3, 25, 5, 72, 6, 51, 7, 81, 8, 58, 9, 15],
    [2, 30, 1, 35, 0, 27, 4, 82, 3, 44, 7, 92, 6, 25, 5, 49, 9, 28, 8, 77],
    [1, 53, 0, 83, 4, 73, 3, 26, 2, 77, 6, 33, 5, 92, 9, 99, 8, 38, 7, 38],
    [1, 20, 0, 44, 4, 81, 3, 88, 2, 66, 6, 70, 5, 91, 9, 37, 8, 55, 7, 96],
    [1, 21, 2, 93, 4, 22, 0, 56, 3, 34, 6, 40, 7, 53, 9, 46, 5, 29, 8, 63],
    [1, 32, 2, 63, 4, 36, 0, 26, 3, 17, 5, 85, 7, 15, 8, 55, 9, 16, 6, 82],
    [0, 73, 2, 46, 3, 89, 4, 24, 1, 99, 6, 92, 7, 7, 9, 51, 5, 19, 8, 14],
    [0, 52, 2, 20, 3, 70, 4, 98, 1, 23, 5, 15, 7, 81, 8, 71, 9, 24, 6, 81]
]
Orb4C = [
    [0, 8, 1, 10, 2, 35, 3, 44, 4, 15, 5, 92, 6, 70, 7, 89, 8, 50, 9, 12],
    [0, 63, 8, 39, 3, 80, 5, 22, 2, 88, 1, 39, 9, 85, 6, 27, 7, 74, 4, 69],
    [0, 52, 6, 22, 1, 33, 3, 68, 8, 27, 2, 68, 5, 25, 4, 34, 7, 24, 9, 84],
    [0, 31, 1, 85, 4, 55, 8, 80, 5, 58, 7, 11, 6, 69, 9, 56, 3, 73, 2, 25],
    [0, 97, 5, 98, 9, 87, 8, 47, 7, 77, 4, 90, 3, 98, 2, 80, 1, 39, 6, 40],
    [1, 97, 5, 68, 0, 44, 9, 67, 2, 44, 8, 85, 3, 78, 6, 90, 7, 33, 4, 81],
    [0, 34, 3, 76, 8, 48, 7, 61, 9, 11, 2, 36, 4, 33, 6, 98, 1, 7, 5, 44],
    [0, 44, 9, 5, 4, 85, 1, 51, 5, 58, 7, 79, 2, 95, 6, 48, 3, 86, 8, 73],
    [0, 24, 1, 63, 9, 48, 7, 77, 8, 73, 6, 74, 4, 63, 5, 17, 2, 93, 3, 84],
    [0, 51, 2, 5, 4, 40, 9, 60, 1, 46, 5, 58, 8, 54, 3, 72, 6, 29, 7, 94]
]
Orb5C = [
    [9, 11, 8, 93, 0, 48, 7, 76, 6, 13, 5, 71, 3, 59, 2, 90, 4, 10, 1, 65],
    [8, 52, 9, 76, 0, 84, 7, 73, 5, 56, 4, 10, 6, 26, 2, 43, 3, 39, 1, 49],
    [9, 28, 8, 44, 7, 26, 6, 66, 4, 68, 5, 74, 3, 27, 2, 14, 1, 6, 0, 21],
    [0, 18, 1, 58, 3, 62, 2, 46, 6, 25, 4, 6, 5, 60, 7, 28, 8, 80, 9, 30],
    [0, 78, 1, 47, 7, 29, 5, 16, 4, 29, 6, 57, 3, 78, 2, 87, 8, 39, 9, 73],
    [9, 66, 8, 51, 3, 12, 7, 64, 5, 67, 4, 15, 6, 66, 2, 26, 1, 20, 0, 98],
    [8, 23, 9, 76, 6, 45, 7, 75, 5, 24, 3, 18, 4, 83, 2, 15, 1, 88, 0, 17],
    [9, 56, 8, 83, 7, 80, 6, 16, 4, 31, 5, 93, 3, 30, 2, 29, 1, 66, 0, 28],
    [9, 79, 8, 69, 2, 82, 4, 16, 5, 62, 3, 41, 6, 91, 7, 35, 0, 34, 1, 75],
    [0, 5, 1, 19, 2, 20, 3, 12, 4, 94, 5, 60, 6, 99, 7, 31, 8, 96, 9, 63]
]
Orb6C = [
    [0, 99, 1, 74, 2, 49, 3, 67, 4, 17, 5, 7, 6, 9, 7, 39, 8, 35, 9, 49],
    [0, 49, 3, 67, 4, 82, 2, 92, 1, 62, 5, 84, 8, 45, 6, 30, 7, 42, 9, 71],
    [0, 26, 3, 33, 1, 82, 2, 98, 5, 83, 4, 16, 6, 64, 7, 65, 9, 36, 8, 77],
    [0, 41, 1, 62, 4, 73, 3, 94, 6, 51, 5, 46, 2, 55, 9, 31, 7, 64, 8, 46],
    [1, 68, 0, 26, 5, 50, 3, 46, 4, 25, 7, 88, 2, 6, 8, 13, 9, 98, 6, 84],
    [0, 24, 6, 80, 2, 91, 3, 55, 1, 48, 8, 99, 4, 72, 9, 91, 7, 84, 5, 12],
    [2, 16, 3, 13, 0, 9, 1, 58, 4, 23, 7, 85, 5, 36, 6, 89, 8, 71, 9, 41],
    [2, 54, 0, 41, 3, 38, 4, 53, 1, 11, 5, 74, 9, 88, 6, 46, 7, 41, 8, 65],
    [2, 53, 1, 50, 4, 40, 0, 90, 7, 7, 5, 80, 3, 57, 9, 60, 6, 91, 8, 47],
    [2, 45, 0, 59, 8, 81, 3, 99, 6, 71, 1, 19, 4, 75, 7, 77, 9, 94, 5, 95]
]
Orb7C = [
    [0, 32, 1, 14, 2, 15, 3, 37, 4, 18, 5, 43, 6, 19, 7, 27, 8, 28, 9, 31],
    [0, 8, 3, 12, 4, 49, 8, 24, 9, 52, 6, 19, 7, 23, 5, 19, 2, 17, 1, 32],
    [0, 25, 7, 19, 3, 27, 2, 45, 6, 21, 4, 15, 1, 13, 5, 16, 8, 43, 9, 19],
    [0, 24, 1, 18, 3, 41, 8, 29, 5, 14, 2, 17, 4, 23, 7, 15, 6, 18, 9, 23],
    [0, 27, 6, 29, 1, 39, 3, 21, 5, 15, 2, 15, 9, 25, 7, 26, 8, 44, 4, 20],
    [4, 17, 0, 15, 6, 51, 8, 17, 2, 46, 3, 16, 9, 33, 7, 25, 5, 30, 1, 25],
    [4, 15, 3, 31, 0, 25, 9, 12, 2, 13, 1, 51, 8, 19, 7, 21, 6, 12, 5, 26],
    [4, 8, 6, 29, 9, 25, 3, 15, 8, 17, 2, 22, 7, 32, 5, 20, 0, 11, 1, 28],
    [2, 41, 6, 10, 8, 32, 7, 5, 4, 21, 9, 59, 3, 26, 1, 10, 5, 16, 0, 29],
    [2, 20, 9, 7, 5, 44, 8, 22, 6, 33, 3, 25, 7, 29, 4, 12, 1, 14, 0, 0]
]
Orb8C = [
    [0, 55, 1, 74, 2, 45, 3, 23, 4, 76, 5, 19, 6, 18, 7, 61, 8, 44, 9, 11],
    [0, 63, 1, 43, 2, 51, 3, 18, 4, 42, 7, 11, 6, 29, 5, 52, 9, 29, 8, 88],
    [2, 88, 1, 31, 0, 47, 4, 10, 3, 62, 5, 60, 6, 58, 7, 29, 8, 52, 9, 92],
    [2, 16, 1, 71, 0, 55, 4, 55, 3, 9, 7, 49, 6, 83, 5, 54, 9, 7, 8, 57],
    [1, 7, 0, 41, 4, 92, 3, 94, 2, 46, 6, 79, 5, 34, 9, 38, 8, 8, 7, 18],
    [1, 25, 0, 5, 4, 89, 3, 94, 2, 14, 6, 94, 5, 20, 9, 23, 8, 44, 7, 39],
    [1, 24, 2, 21, 4, 47, 0, 40, 3, 94, 6, 71, 7, 89, 9, 75, 5, 97, 8, 15],
    [1, 5, 2, 7, 4, 74, 0, 28, 3, 72, 5, 61, 7, 9, 8, 53, 9, 32, 6, 97],
    [0, 34, 2, 52, 3, 37, 4, 6, 1, 94, 6, 6, 7, 56, 9, 41, 5, 5, 8, 16],
    [0, 77, 2, 74, 3, 82, 4, 10, 1, 29, 5, 15, 7, 51, 8, 65, 9, 37, 6, 21]
]
#基础信息
class Data():
    def __init__(self , data_list = Orb7C):
        self.data_list = data_list
        self.kind_task_tuple = list((p, j) for p in range(10) for j in range(10))
        self.kind_task_tuple.remove((9, 9))
        self.kind_task_tuple = tuple(self.kind_task_tuple)
        self.kind_tuple = tuple(i for i in range(10))
        self.kind_r_tuple = (3,4,5,7,8,9)
        self.machine_count = 10
        self.machine_tuple = tuple(m for m in range(self.machine_count))
        self.task_p_dict = {p: tuple(j for j in range(10)) for p in range(10)}
        self.machine_pj_dict = {(p,j):tuple([self.data_list[p][2*j]]) for (p,j) in self.kind_task_tuple}
        self.time_pjm_dict = {(p,j):{m:self.data_list[p][2*j+1] for m in self.machine_pj_dict[(p,j)]} for (p,j) in self.kind_task_tuple}
        self.time_mpj_dict = {m: {(p, j): self.time_pjm_dict[(p, j)][m] for (p, j) in self.kind_task_tuple if m in self.machine_pj_dict[(p, j)]}for m in self.machine_tuple}
        self.assemble_task = [(0,0),(1,0),(2,0),(6,0)]

        self.cost_pj_dict = {(p,j):[1] for (p,j) in self.kind_task_tuple}
        for (p,j) in self.assemble_task:
            self.cost_pj_dict[(p,j)] = [1,1]
        self.cost_pj_dict[(0,0)] = [1,1,1]

        self.pre_pj_dict = {(p, j) : [(p,j-1)] for (p,j) in self.kind_task_tuple}
        for (p,j) in self.kind_task_tuple:
            if j == 0:
                self.pre_pj_dict[(p,j)] = []
        self.pre_pj_dict[(0,0)] = [(1,9),(2,9),(3,9)]
        self.pre_pj_dict[(1, 0)] =[(4,9),(5,9)]
        self.pre_pj_dict[(2, 0)] = [(6, 9), (7, 9)]
        self.pre_pj_dict[(6, 0)] = [(8, 9), (9, 8)]

        self.total_cost_dict = {(p,j): 1 for (p,j) in self.kind_task_tuple}
        self.cost_count = {(p,j):[self.pre_pj_dict[(p,j)],self.cost_pj_dict[(p,j)]] for (p,j) in self.kind_task_tuple}

        self.post_pj_dict = {(p, j): [] for (p, j) in self.kind_task_tuple}  # 初始化为空列表
        for current_pj, pre_list in self.pre_pj_dict.items():
            for pre_pj in pre_list:
                if pre_pj in self.post_pj_dict:
                    self.post_pj_dict[pre_pj].append(current_pj)

        self.kind_task_m_dict = {m: tuple((p, j) for (p, j) in self.kind_task_tuple if m in self.machine_pj_dict[(p, j)]) for m in
                            self.machine_tuple}

        self.time_pj_dict = {
            (p, j): sum([self.time_pjm_dict[(p, j)][m] for m in self.machine_pj_dict[(p, j)]]) / len(self.machine_pj_dict[(p, j)])
            for (p, j) in self.kind_task_tuple}

"""流体模型"""
class Part():
    """
    定义部件类相当于定义工序o_vj类
    p是o_vj的索引
    """
    def __init__(self,p):
        self.part = p
        self.part_unprocess_list = []
        #基本属性
        self.task = p #索引
        self.machine_tuple = None  # 可选加工机器编号
        self.time_remain = None  # 后续剩余处理时间
        # 附加属性
        self.buffer = None #该工序段已经完成的工序的缓冲区容量,选择才减少
        self.count = None #该工序段已经完成的工序的容量，分配了才减少，主要针对交叉混用的装配工序
        self.part_now_list = []  # 处于该工序段的工件对象列表
        self.part_unprocessed_list = []  # 该工序段未被加工的工件对象列表同时也是未被加工的“工序”对象列表
        self.part_unselected_list = []  #该工序段可转移到part_now_list的对象
        self.part_processed_list = []  # 该工序段已加工的工序对象列表同时也是已加工的“工序“对象列表
        # 流体相关属性
        self.fluid_process_rate_m_dict = {}  # 流体中被各机器加工的速率
        self.fluid_machine_list = []  # 流体模型解可选加工机器
        self.fluid_time_sum = None  # 流体模型中该工序的加工时间
        self.fluid_rate_sum = None  # 流体模型中加工该工序的速率
        self.fluid_number = None  # 处于该工序段的流体数量
        self.fluid_unprocessed_number = None  # 未被加工的流体数

class Job(Part):
    """工件类"""
    def __init__(self, p, n):   #后续动态进行调整终端工件
        Part.__init__(self, p)  # 调用父类的构函
        # 基本属性
        self.number = n  # 该工件类型的第几个工件或者工件编号
        self.time_arrive = None  # 该工件的到达时间
        # 附加属性
        self.part_list = []  # 分配机器的工件对象列表
        self.part_unprocessed_list = []  # 未分配机器的“工序”对象列表或者未分配机器的工件对象列表

class Task(Job):
    """工序类"""
    def __init__(self, p, n):
        Job.__init__(self, p, n)  # 调用父类构函
        # 附加属性
        self.machine = None  # 选择的机器
        self.time_end = None  # 加工结束时间
        self.time_begin = None  # 加工开始时间
        """加工耗时"""
        @property
        def time_cost(self):
            return self.time_end - self.time_begin

class Machine():
    """机器类"""
    def __init__(self, m):
        # 基本属性
        self.machine_node = m  # 机器编号
        self.kind_task_tuple = None  # 可选加工工序类型元组
        # 附加属性
        self.state = 0  # 机器状态
        self.time_end = 0  # 机器完工时间
        self.part_list = []  # 机器已加工“工序”对象列表 机器已加工工件对象列表
        self.part_object = None  # 机器正在处理的工件对象
        self.unprocessed_part_dict = {}  # 未被m加工的各工件类型的工件总数/随着加工过程动态更新
        # 流体附加属性
        self.fluid_part_list = []  # 可选加工工件类型
        self.time_ratio_part_dict = {}  # 流体解中分配给各工序类型的时间比例
        self.fluid_process_rate_part_dict = {}  # 流体解中加工各工件类型的速率
        self.fluid_unprocessed_part_dict = {}  # 未被机器m加工的各工件类型流体总数

class AJSP(Data):
    """柔性装配作业车间调度类"""
    def __init__(self, **kwargs):
        Data.__init__(self, **kwargs)  # 调用父类的构函
        # 实例化工件类型、工件、工序类型、工序和机器对象字
        self.fluid_completed_time = None #流体完工时间
        self.part_dict = {p:Part(p) for p in range(len(self.kind_task_tuple))} #工件（工序）对象列表（索引值表示）
        self.machine_dict = {m: Machine(m) for m in self.machine_tuple}  # 机器对象字典
        self.task_dict = {}  # (r,n)  # 工件对象字典
        self.process_rate_m_part_dict = {
            m: {(v, j): 1 / self.time_mpj_dict[m][(v, j)] for (v, j) in self.kind_task_m_dict[m]} for m in
            self.machine_tuple}  # 机器加工流体速率
        self.reset_parameter()
        # self.reset_object()
        # print("成功定义FAJSP类")

    def reset_parameter(self):
        """初始化各字典和参数"""
        for p,part_object in self.part_dict.items():
            (v,j) = self.kind_task_tuple[p] #工件p对应的工序对象
            part_object.part_unprocess_list = [] #未分配机器的工件对象列表？？
            part_object.machine_tuple = self.machine_pj_dict[(v,j)]  # 可选加工机器编号元组
            # part_object.time_remain = sum(self.time_pj_dict[(v, j)] for j in self.task_p_dict[v][j:])  # 后续剩余处理时间
            part_object.job_now_list = []  # 处于该工序段的工件对象列表
            part_object.job_unprocessed_list = []  # 该工序段未被加工的工件对象列表
            part_object.task_processed_list = []  # 该工序段已加工的工序对象列表
        for m, machine_object in self.machine_dict.items():
            machine_object.kind_task_tuple = self.kind_task_m_dict[m]  # 可选加工工序类型元组
            machine_object.machine_state = 0  # 机器状态
            machine_object.time_end = 0  # 机器完工时间
            machine_object.part_list = []  # 机器已加工工序对象列表
            machine_object.job_object = None


    def reset_fluid_parameter(self):
        """初始化流体属性参数"""
        for part, part_object in self.part_dict.items():
            part_object.fluid_machine_list = []  # 流体模型中可选加工机器
            part_object.fluid_process_rate_m_dict = {}  # 被各机器加工的速率
        for m, machine_object in self.machine_dict.items():
            machine_object.fluid_part_list = []  # 流体解中可选加工工序类型列表
            machine_object.time_ratio_part_dict = {}  # 流体解中分配给各工序类型的时间比例
            machine_object.fluid_process_rate_part_dict = {}  # 流体解中加工各工序类型的速率
            machine_object.unprocessed_part_dict = {}  # 未被m加工的各工件的总数 (p,j)
            machine_object.fluid_unprocessed_part_dict = {}  # 流体解中未被机器m加工的各工序类型总数

    def reset_object(self):                    #重写该方法
        """
        添加工件对象和工序对象+更新流体模型和属性
        :return: 添加工序对象和机器对象+更新流体模型和属性
        """
        # 清理之前创建的工件对象
        self.task_dict.clear()
        for p in self.part_dict.keys():
            self.part_dict[p].part_now_list.clear()
            self.part_dict[p].part_unselected_list.clear()
            self.part_dict[p].part_unprocessed_list.clear()
            self.part_dict[p].part_processed_list.clear()
            self.part_dict[p].buffer = 0
            self.part_dict[p].count = 0

        for p in self.part_dict.keys():
            (v,j) = self.kind_task_tuple[p]
            # batch_size = self.total_cost_dict[(v, j)]
            # print(f"工序索引 p={p}, 工序(v,j)=({v},{j}), 批量大小={batch_size}")
            n_start = 0
            n_end = n_start + self.total_cost_dict[(v,j)]
            # print(f"  创建 {n_end - n_start} 个工序工件")
            self.part_dict[p].buffer = 0
            self.part_dict[p].count = 0

            for n in range(n_start, n_end):
                part_object =  Task(p,n) #实例化工件对象
                part_object.part_list = []
                part_object.part_unprocessed_list = []
                if v in self.kind_r_tuple and j == 0:
                    idx = self.kind_task_tuple.index((v,j))
                    self.part_dict[idx].part_now_list.append(part_object)  # 初始时刻只有源工件有待加工工件
                else:
                    idx = self.kind_task_tuple.index((v, j))
                    self.part_dict[idx].part_unselected_list.append(part_object)
                # part_object.part_unprocessed_list.append(part_object)
                self.part_dict[p].part_unprocessed_list.append(part_object)  # 未完成工件对象列表
                self.task_dict[(p, n)] = part_object  # 加入工序字典
            # 初始化流体属性
        for p in self.part_dict.keys():
            self.part_dict[p].fluid_number = len(self.part_dict[p].part_now_list)  # 处于该工序段的流体数量
            self.part_dict[p].fluid_unprocessed_number = len(self.part_dict[p].part_unprocessed_list)  # 未被加工的流体数
        x = self.fluid_model()
        # 初始化流体属性
        self.reset_fluid_parameter()
        # 基于流体解更新流体属性
        self.update_fluid_parameter(x)

    def fluid_model(self):
        """
        最小化最大完工时间目标流体模型求解
        """
        # 初始化模型对象
        model = Model('LP')
        # 定义决策变量
        var_list = {(m, (v, j)) for m in self.machine_tuple for (v, j) in self.kind_task_m_dict[m]}
        X = model.continuous_var_dict(var_list, lb=0, ub=1, name='X')
        # 各流体初始未加工数量
        fluid_number = {(v,j):self.part_dict[self.kind_task_tuple.index((v,j))].fluid_unprocessed_number for (v,j) in self.kind_task_tuple}
        # 各流体初始瞬态数量
        fluid_number_time = {(v, j): self.part_dict[self.kind_task_tuple.index((v,j))].fluid_number for (v,j) in self.kind_task_tuple}
        process_rate_part_sum = {(v, j): sum(X[m, (v, j)] * self.process_rate_m_part_dict[m][(v, j)]
                                           for m in self.machine_pj_dict[(v, j)]) for (v, j) in self.kind_task_tuple}
        # 定义目标函数
        model.maximize(model.min(process_rate_part_sum[(v, j)] / fluid_number[(v, j)] for (v, j) in self.kind_task_tuple))
        # 机器利用率约束
        model.add_constraints(model.sum(X[m, (v, j)] for (v, j) in self.kind_task_m_dict[m]) <= 1
                                  for m in self.machine_tuple)
        # 解的可行性约束 (V,J)表示当前工序， (v,j)表示前置工序
        for (V, J) in self.kind_task_tuple:
            if fluid_number_time[(V, J)] == 0 and self.pre_pj_dict[(V, J)]:
                for (v, j) in self.pre_pj_dict[(V, J)]:
                    model.add_constraint(process_rate_part_sum[(v, j)] >= process_rate_part_sum[(V, J)])
        # 求解模型
        solution = model.solve()
        x = solution.get_value_dict(X)
        # 输出流体完工时间
        process_rate_pj_sum = {(v, j): sum(x[m, (v, j)] * self.process_rate_m_part_dict[m][(v, j)]
                                           for m in self.machine_pj_dict[(v, j)]) for (v, j) in self.kind_task_tuple}
        self.fluid_completed_time = max(
            fluid_number[(v, j)] / process_rate_pj_sum[(v, j)] for (v, j) in self.kind_task_tuple)
        #
        # print("流体完工时间：", self.fluid_completed_time)
        # print(solution)
        return x



    def update_fluid_parameter(self, x):
            """基于流体解更新流体参数"""
            for (m, (v, j)), rate in x.items():
                p = self.kind_task_tuple.index((v,j))
                machine_object = self.machine_dict[m]
                part_object = self.part_dict[p]
                machine_object.time_ratio_part_dict[(v, j)] = rate  # 流体解中分配给各工序类型的时间比例
                part_object.fluid_process_rate_m_dict[m] = rate * self.process_rate_m_part_dict[m][(v, j)]
                machine_object.fluid_process_rate_part_dict[(v, j)] = rate * self.process_rate_m_part_dict[m][(v, j)]
                if rate != 0:
                    machine_object.fluid_part_list.append((v, j))
                    part_object.fluid_machine_list.append(m)
            for p, part_object in self.part_dict.items():
                part_object.fluid_rate_sum = sum(part_object.fluid_process_rate_m_dict.values())  # 工序类型处理速率
                part_object.fluid_time_sum = 1 / part_object.fluid_rate_sum  # 工序类型的加工时间
            for m, machine_object in self.machine_dict.items():
                for (v, j) in machine_object.kind_task_tuple:  # 添加除流体模型中可选工序类型外的工序
                    p = self.kind_task_tuple.index((v, j))
                    part_object = self.part_dict[p]
                    # 未被m加工的各工序类型流体数量
                    machine_object.fluid_unprocessed_part_dict[(v, j)] = \
                        part_object.fluid_unprocessed_number * \
                        machine_object.fluid_process_rate_part_dict[(v, j)] / part_object.fluid_rate_sum
                    # 未被m加工的工序o_pj的总数 (p,j)
                    machine_object.unprocessed_part_dict[(v, j)] = machine_object.fluid_unprocessed_part_dict[
                        (v, j)]
                    # 流体解中未被机器m加工的各工序类型总数
                    machine_object.fluid_unprocessed_part_dict[(p, j)] = machine_object.fluid_unprocessed_part_dict[
                        (v, j)]

"""AJSP环境"""
class AJSP_Environment(AJSP):
    def __init__(self,**kwargs):
        super().__init__( **kwargs)
        self.step_count = 0 #决策步
        self.last_step_time = 0
        self.step_time = 0 #决策时间点
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        self.completion_time = 0  # 当前时间点完工时间
        self.completion_time_last = 0  # 上一时间步完工时间
        #状态，动作属性
        self.current_state = None # 当前状态
        self.next_state = None  # 下一步状态
        self.actions = ((p, j, m) for m in self.machine_tuple for (p, j) in self.kind_task_tuple if m in self.kind_task_m_dict[(p, j)])  # 动作空间
        self.reward = 0 #累计回报
        self.reward_sum =None
        self.fluid_solution = None
        #记录甘特图
        self.schedule_log = []

    def reset(self):
        """环境重置状态"""
        self.reset_parameter() #初始化参数对象列表和字典
        self.reset_object()
        self.fluid_solution = self.fluid_model()
        self.update_fluid_parameter(self.fluid_solution)
        self.reward_sum = 0  # 累计回报
        # 重置时间步和当前时间点
        self.step_count = 0
        self.last_step_time = 0
        self.step_time = 0
        self.completion_time = 0  # 当前时间点完工时间
        self.completion_time_last = 0  # 上一时间步完工时间
        #重置状态属性
        self.current_state = self.state_extract()  # 当前状态
        self.next_state = None  # 下一步状态
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        #记录甘特图
        self.schedule_log = []

    def state_extract(self):
        """状态提取函数"""
        #初始化状态矩阵
        proc_rate_matrix = np.zeros((self.machine_count, len(self.kind_task_tuple))) # 实际加工速率矩阵
        job_flow_matrix = np.zeros((len(self.kind_task_tuple), len(self.kind_task_tuple)))  # 物料工序流矩阵
        fluid_rate_matrix =  np.zeros((self.machine_count, len(self.kind_task_tuple)))# 流体加工速率矩阵
        pend_job_matrix = np.zeros((3, len(self.kind_task_tuple)))  # 待加工物料矩阵

        #静态特征
        for m in self.machine_tuple:
            for (p, j) in self.time_mpj_dict[m].keys():
                proc_rate_matrix[m][self.kind_task_tuple.index((p, j))] = 1/self.time_mpj_dict[m][(p, j)]

        for j_pre in self.kind_task_tuple:
            for j_cur in self.kind_task_tuple:
                if j_pre in self.pre_pj_dict[j_cur]:
                    idx_j_pre = self.kind_task_tuple.index(j_pre)
                    idx_j_cur = self.kind_task_tuple.index(j_cur)
                    idx_j_pre_cost = self.pre_pj_dict[j_cur].index(j_pre)
                    job_flow_matrix[idx_j_pre][idx_j_cur] = self.cost_pj_dict[j_cur][idx_j_pre_cost ]

        # 基于流体模型在线计算的动态特征
        for (m,(p,j)),rate in self.fluid_solution.items():
            idx = self.kind_task_tuple.index((p, j))
            fluid_rate_matrix[m][idx] = rate

        # 待处理物料矩阵
        for (p, j) in self.kind_task_tuple:
            idx = self.kind_task_tuple.index((p, j))
            part_object = self.part_dict[idx]

            pend_job_matrix[0][idx] = self.total_cost_dict[(p, j)]  # 初始各工序待加工数量
            pend_job_matrix[1][idx] = len(part_object.part_unprocessed_list)  # 当前未处理工序数
            pend_job_matrix[2][idx] = part_object.fluid_unprocessed_number  # 流体模型中未处理数量
        # print(pend_job_matrix)

        return [proc_rate_matrix, job_flow_matrix, fluid_rate_matrix, pend_job_matrix]


    """
    代办,根据count选候选动作
    """
    def get_candidate_actions(self):
        """构建当前决策点的候选动作集"""
        candidate_actions = []
        for p,part_object in self.part_dict.items():
            (v,j) = self.kind_task_tuple[p]
            # 检查1: 必须有待处理工件（关键修复）
            if not part_object.part_unprocessed_list:
                continue  # 如果没有待处理工件，直接跳过
            pre_vj  = [pre_vj for pre_vj in self.pre_pj_dict[(v,j)]]    #该工序的前置工序
            if not pre_vj:
                for m in self.machine_pj_dict[(v,j)]:
                    if len(part_object.part_now_list) >= 1 and self.machine_dict[m].state == 0:
                        candidate_actions.append((v,j,m))
            if len(pre_vj) == 1:
                idx = self.kind_task_tuple.index(pre_vj[0])
                for m in self.machine_pj_dict[(v,j)]:
                    if self.part_dict[idx].count >= 1 and self.machine_dict[m].state == 0:
                        candidate_actions.append((v,j,m))
            if len(pre_vj) == 2:
                pre_idx_0 = self.pre_pj_dict[(v,j)].index(pre_vj[0])
                pre_idx_1 = self.pre_pj_dict[(v,j)].index(pre_vj[1])
                idx_0 = self.kind_task_tuple.index(pre_vj[0])
                idx_1 = self.kind_task_tuple.index(pre_vj[1])
                for m in self.machine_pj_dict[(v, j)]:
                    if self.part_dict[idx_0].count >= self.cost_count[(v,j)][1][pre_idx_0] and self.part_dict[idx_1].count >= self.cost_count[(v,j)][1][pre_idx_1] and self.machine_dict[m].state == 0:
                        candidate_actions.append((v, j, m))
            if len(pre_vj) == 3:
                pre_idx_0 = self.pre_pj_dict[(v, j)].index(pre_vj[0])
                pre_idx_1 = self.pre_pj_dict[(v, j)].index(pre_vj[1])
                pre_idx_2 = self.pre_pj_dict[(v, j)].index(pre_vj[2])
                idx_0 = self.kind_task_tuple.index(pre_vj[0])
                idx_1 = self.kind_task_tuple.index(pre_vj[1])
                idx_2 = self.kind_task_tuple.index(pre_vj[2])
                for m in self.machine_pj_dict[(v, j)]:
                    if (self.part_dict[idx_0].count >= self.cost_count[(v, j)][1][pre_idx_0] and
                            self.part_dict[idx_1].count >= self.cost_count[(v, j)][1][pre_idx_1] and
                            self.part_dict[idx_2].count >= self.cost_count[(v, j)][1][pre_idx_2] and
                            self.machine_dict[m].state == 0):
                        candidate_actions.append((v, j, m))
        return candidate_actions


    def action_features(self,action):
        """构建动作特征向量"""
        vj = (action[0], action[1])  # 选择的工序
        m = action[2]  # 选择的机器
        [PR_vjm,MFN_jj,FPR_vjm,PM_vj] = self.state_extract()
        idx = self.kind_task_tuple.index((vj))
        rate_vjm = [PR_vjm[m][idx]]
        fluid_rate_vjm = [FPR_vjm[m][idx]]
        flow_matrix = [i for i in MFN_jj[:,idx]]
        pend_matrix = [i for i in PM_vj[:,idx]]
        feature_vector = {action: rate_vjm+fluid_rate_vjm+flow_matrix+pend_matrix}
        # print(feature_vector)
        return feature_vector

    def step(self,action):
        """环境执行一步动作"""
        #执行动作 action = (p,j,m))
        (v,j,m) = action
        vj_selected = (v,j)
        m_selected = m
        p = self.kind_task_tuple.index(vj_selected)
        kind_task_object_selected = self.part_dict[p]  # 工序对象
        part_object_selected = kind_task_object_selected.part_unprocessed_list[0] #待加工工件
        machine_object_selected = self.machine_dict[m_selected]  # 机器对象
        #更新工件对象属性
        part_object_selected.time_begin = self.step_time #工序开工时间
        part_object_selected.machine = m_selected #选择的机器
        part_object_selected.time_end = self.step_time + self.time_mpj_dict[m_selected][vj_selected]  # 完工时间
        kind_task_object_selected.part_unprocessed_list.remove(part_object_selected) #更新未分配机器的工件对象列表
        #甘特图日志
        self.schedule_log.append({
            "number":part_object_selected.number,
            "job": v,
            "op": j,
            "machine": m_selected,
            "start": part_object_selected.time_begin,
            "end": part_object_selected.time_end
        })
        #更新count属性和part_now_list属性
        if not self.pre_pj_dict[(v, j)]:  # 前置工序列表为空
            if kind_task_object_selected.part_now_list:
                del kind_task_object_selected.part_now_list[0]
        else:  # 有前置工序
            for pre_job in self.pre_pj_dict[(v, j)]:
                if pre_job:  # 如果前置工序存在
                    pre_job_idx = self.kind_task_tuple.index(pre_job)
                    pre_idx = self.pre_pj_dict[(v, j)].index(pre_job)
                    self.part_dict[pre_job_idx].count -= self.cost_count[(v, j)][1][pre_idx]
        #更新机器对象
        machine_object_selected.state = 1  # 更新机器状态
        machine_object_selected.time_end = part_object_selected.time_end  # 更新机器完工时间
        machine_object_selected.part_list.append(part_object_selected)  # 更新机器已加工工序对象列表
        machine_object_selected.job_object = part_object_selected  # 更新机器正在加工的工件对象
        machine_object_selected.unprocessed_part_dict[vj_selected] -= 1  # 更新该机器未加工工序pj的数量，这个属性由流体模型确定，后续再更新流体模型参数就会影响这个值，后续手动更新流体模型参数
        self.completion_time = max(self.completion_time, part_object_selected.time_end)
        #判断是否移动时间钟
        while len(self.get_candidate_actions()) == 0:
            time_point_list = [self.machine_dict[m].time_end for m in self.machine_tuple
                               if self.machine_dict[m].time_end > self.step_time]
            self.step_time = min(time_point_list)
            # print(self.step_time)
            #更新工件对象的缓冲区容量，完成当前工件则count+1
            for m, machine_object in self.machine_dict.items():
                if machine_object.time_end == self.step_time:
                    job_object = machine_object.job_object
                    p=job_object.part #工件索引
                    machine_object.state = 0
                    self.part_dict[p].count += 1     #一个工序完成计数+1
            # 更新机器状态和正在加工的工件对象
            for m, machine_object in self.machine_dict.items():
                if machine_object.time_end <= self.step_time:
                    machine_object.state = 0  # 更新机器状态
            #更新流体属性
            for p, part_object in self.part_dict.items():
                (v,j) = self.kind_task_tuple[p]
                part_object.fluid_unprocessed_number = max(0,self.total_cost_dict[(v, j)] - part_object.fluid_rate_sum * self.step_time)
            # 判断是否终止
            if sum(len(part_object.part_unprocessed_list) for p, part_object in self.part_dict.items()) == 0:
                self.done = True
                break
        # 提取新的状态、计算回报值、判断周期循环是否结束
        # 更新流体更新时间
        self.step_count += 1
        self.last_step_time = self.step_time
        self.reward = self.compute_reward()  # 即时奖励
        self.reward_sum += self.reward  # 更新累计回报
        self.completion_time_last = self.completion_time
        self.current_state = self.state_extract()  # 提取当前状态
        # print(
        #     f"步骤 {self.step_count}: 调度 {action}, 剩余未处理: {len(kind_task_object_selected.part_unprocessed_list)}")
        return self.current_state, self.reward, self.done



    def compute_reward(self):
        """计算即时奖励"""
        return -(self.completion_time - self.completion_time_last)/self.fluid_completed_time
"""卷积神经网络"""
class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器 - 用于提取状态矩阵的空间特征"""
    def __init__(self, input_channels=1):
        super(CNNFeatureExtractor, self).__init__()

        # 卷积层序列
        self.conv_layers = nn.Sequential(
            # 第一层卷积 - 增加通道数
            nn.Conv2d(input_channels, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            # nn.AvgPool2d((2, 2)),
            # nn.Dropout(0.1),  # 添加Dropout

            # # 第二层卷积 - 增加通道数
            # nn.Conv2d(32, 64, kernel_size=2, padding=1),
            # nn.ReLU(),
            # nn.AvgPool2d((2, 2)),
            # nn.Dropout(0.1),

            # # 第三层卷积 - 新增
            # nn.Conv2d(64, 128, kernel_size=2, padding=1),
            # nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            # nn.Dropout(0.1)
        )


    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(conv_features.size(0), -1)
        return conv_features

"""自注意力机制"""
class SelfAttention(nn.Module):
    """自注意力机制 - 用于理解动作特征间的关系"""

    def __init__(self, feature_dim = None, num_heads=1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim

        # 线性变换层
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x shape: (batch_size, num_actions, feature_dim)
        batch_size, num_actions, feature_dim = x.shape

        # 线性变换
        Q = self.query(x).view(batch_size, num_actions, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, num_actions, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, num_actions, self.num_heads, self.head_dim)

        # 转置以便矩阵乘法
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        # 应用注意力权重
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_actions, feature_dim)

        # 最终线性变换
        out = self.fc_out(out)
        return out

"""多层感知机"""
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            # nn.Dropout(0.1),
            #
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(0.1),

            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

"""PPO 训练"""
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

#########################################################
# 策略网络：CNN 状态特征 + ATN 动作特征 + MLP 得分
#########################################################
class PolicyNetwork(nn.Module):
    def __init__(self, action_feature_dim, combined_dim):
        super(PolicyNetwork, self).__init__()
        self.device = device

        self.cnn = CNNFeatureExtractor(input_channels=1).to(self.device)
        self.atn = SelfAttention(feature_dim=action_feature_dim, num_heads=1).to(self.device)
        self.mlp = MLP(input_dim=combined_dim).to(self.device)

    def forward(self, matrix_states, candidate_actions, candidate_actions_features):
        # CNN state features
        state_features = []
        for matrix in matrix_states:
            m = torch.tensor(matrix).unsqueeze(0).unsqueeze(0).float().to(self.device)
            feat = self.cnn(m)
            state_features.append(feat)
        state_feature = torch.cat(state_features, dim=1)

        # ATN action features
        action_feature_list = []
        for a in candidate_actions:
            action_feature_list.append(candidate_actions_features[a])
        action_tensor = torch.tensor(action_feature_list).unsqueeze(0).float().to(self.device)
        action_trans = self.atn(action_tensor)

        # Score each action
        scores = []
        for i in range(len(candidate_actions)):
            af = action_trans[:, i, :]
            combined = torch.cat([state_feature, af], dim=1)
            score = self.mlp(combined)
            scores.append(score)

        scores_tensor = torch.cat(scores, dim=1).squeeze(0)
        action_probs = F.softmax(scores_tensor, dim=0)
        return action_probs

#########################################################
# Value 网络
#########################################################
class ValueNetwork(nn.Module):
    def __init__(self, mlp_value_dim):
        super(ValueNetwork, self).__init__()
        self.device = device

        self.cnn = CNNFeatureExtractor(input_channels=1).to(self.device)
        self.mlp = MLP(input_dim=mlp_value_dim).to(self.device)

    def forward(self, matrix_states):
        feats = []
        for matrix in matrix_states:
            m = torch.tensor(matrix).unsqueeze(0).unsqueeze(0).float().to(self.device)
            feat = self.cnn(m)
            feats.append(feat)
        state_feature = torch.cat(feats, dim=1)
        return self.mlp(state_feature)

#########################################################
# PPO Agent
#########################################################
class PPO:
    def __init__(self, action_feature_dim, combined_dim, mlp_value_dim,
                 lr_policy=4e-4, lr_value=5e-5, gamma=0.97,
                 eps_clip=0.25, K_epochs=5, batch_size=24, device=device):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.device = device

        self.policy_net = PolicyNetwork(action_feature_dim, combined_dim).to(self.device)
        self.value_net = ValueNetwork(mlp_value_dim).to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': lr_policy},
            {'params': self.value_net.parameters(), 'lr': lr_value}
        ])

        self.old_policy_net = PolicyNetwork(action_feature_dim, combined_dim).to(self.device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        self.memory = []

        # 添加最优参数记录
        self.best_policy_state = None
        self.best_value_state = None
        self.best_completion_time = float('inf')

        # 添加全局最优参数记录（用于轮训训练）
        self.global_best_policy_state = None
        self.global_best_value_state = None
        self.global_best_avg_ct = float('inf')
        self.global_best_round = -1

    #########################################################
    # 更新最优参数
    #########################################################
    def update_best_params(self, completion_time):
        if completion_time < self.best_completion_time:
            self.best_completion_time = completion_time
            self.best_policy_state = copy.deepcopy(self.policy_net.state_dict())
            self.best_value_state = copy.deepcopy(self.value_net.state_dict())
            return True
        return False

    #########################################################
    # 更新全局最优参数（用于轮训训练）
    #########################################################
    def update_global_best_params(self, avg_completion_time, round_num):
        """更新全局最优参数（基于轮次平均完工时间）"""
        if avg_completion_time < self.global_best_avg_ct:
            self.global_best_avg_ct = avg_completion_time
            self.global_best_policy_state = copy.deepcopy(self.policy_net.state_dict())
            self.global_best_value_state = copy.deepcopy(self.value_net.state_dict())
            self.global_best_round = round_num
            return True
        return False

    #########################################################
    # 选择动作（选择概率最高的动作）
    #########################################################
    def select_action(self, state, candidate_actions, candidate_actions_features):
        with torch.no_grad():
            probs = self.old_policy_net(state, candidate_actions, candidate_actions_features)

            if len(candidate_actions) == 0:
                return None, None, None, None

            # 选择概率最高的动作
            # idx = torch.argmax(probs).item()
            # 依概率采样动作
            idx = torch.multinomial(probs, 1).item()
            action = candidate_actions[idx]
            log_p = torch.log(probs[idx] + 1e-10)

            return action, log_p, probs[idx], candidate_actions_features[action]

    #########################################################
    # 存储经验
    #########################################################
    def store_transition(self, state, action, log_prob,
                         reward, next_state, done,
                         action_features, candidate_actions, candidate_actions_features):

        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob.clone(),
            'reward': float(reward),
            'next_state': next_state,
            'done': float(done),
            'action_features': action_features,
            'candidate_actions': list(candidate_actions),
            'candidate_actions_features': dict(candidate_actions_features)
        })

    #########################################################
    # 计算 Advantage
    #########################################################
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        gamma, lam = self.gamma, 0.95

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * gae * (1 - dones[t])
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    #########################################################
    # PPO 更新
    #########################################################
    def update(self):
        if len(self.memory) < self.batch_size:
            return 0, 0

        states = [m['state'] for m in self.memory]
        actions = [m['action'] for m in self.memory]
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory]).to(self.device)
        rewards = torch.tensor([m['reward'] for m in self.memory]).float().to(self.device)
        dones = torch.tensor([m['done'] for m in self.memory]).float().to(self.device)
        next_states = [m['next_state'] for m in self.memory]

        # 计算 Value
        with torch.no_grad():
            values = torch.tensor([self.value_net(s).item() for s in states]).to(self.device)
            next_values = torch.tensor([
                self.value_net(ns).item() if ns is not None else 0.0 for ns in next_states
            ]).to(self.device)

        advantages = self.compute_advantages(rewards, values, next_values, dones)
        returns = advantages + values

        # 归一化
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses = [], []

        for _ in range(self.K_epochs):
            batch_indices = list(range(len(self.memory)))
            random.shuffle(batch_indices)

            for start in range(0, len(batch_indices), self.batch_size):
                end = start + self.batch_size
                idxs = batch_indices[start:end]
                if len(idxs) == 0:
                    continue

                batch_old = old_log_probs[idxs]
                batch_adv = advantages[idxs]
                batch_ret = returns[idxs]

                new_log_probs = []
                value_preds = []

                for bi, mem_idx in enumerate(idxs):
                    m = self.memory[mem_idx]

                    state = m['state']
                    ca = m['candidate_actions']
                    caf = m['candidate_actions_features']
                    act = m['action']

                    probs = self.policy_net(state, ca, caf)
                    act_index = ca.index(act)
                    new_log = torch.log(probs[act_index] + 1e-10)

                    new_log_probs.append(new_log)
                    value_preds.append(self.value_net(state))

                new_log_probs = torch.stack(new_log_probs).to(self.device)
                value_preds = torch.stack(value_preds).squeeze().to(self.device)

                ratios = torch.exp(new_log_probs - batch_old)

                surr1 = ratios * batch_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value_preds, batch_ret)

                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.memory = []
        return np.mean(policy_losses), np.mean(value_losses)

#########################################################
# Trainer
#########################################################
class Trainer:
    def __init__(self, env, agent, max_episodes=None):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes

        self.episode_rewards = []
        self.completion_times = []
        self.policy_losses = []
        self.value_losses = []

    def train(self):
        print("开始训练...")
        start = time.time()

        best_completion_time = float('inf')
        best_schedule_log = None

        for episode in range(self.max_episodes):
            # print("周期:",episode)
            self.env.reset()
            state = self.env.current_state
            total_reward = 0

            while 1 :
                candidate_actions = self.env.get_candidate_actions()
                if not candidate_actions:
                    break

                candidate_actions_features = {
                    a: self.env.action_features(a)[a]
                    for a in candidate_actions
                }

                action, logp, _, act_feat = \
                    self.agent.select_action(state, candidate_actions, candidate_actions_features)
                # print(action)
                next_state, reward, done = self.env.step(action)

                self.agent.store_transition(
                    state, action, logp, reward, next_state, done,
                    act_feat, candidate_actions, candidate_actions_features
                )

                state = next_state
                total_reward += reward

                if done:
                    break

            pol_loss, val_loss = self.agent.update()

            self.episode_rewards.append(total_reward)
            self.completion_times.append(self.env.completion_time)
            self.policy_losses.append(pol_loss)
            self.value_losses.append(val_loss)
            # ========== 跟踪并保存最优完工时间对应的 schedule_log ==========
            # 如果当前 episode 的完工时间更短，保存该 episode 的调度日志（深拷贝）
            try:
                current_ct = float(self.env.completion_time)
            except Exception:
                current_ct = float(self.env.completion_time)

            if current_ct < best_completion_time:
                best_completion_time = current_ct
                # 深拷贝 schedule_log，防止后续被 env 覆盖
                best_schedule_log = copy.deepcopy(getattr(self.env, "schedule_log", []))

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{self.max_episodes}, "
                      f"Reward={np.mean(self.episode_rewards[-10:]):.3f}, "
                      f"CT={np.mean(self.completion_times[-10:]):.2f}, "
                      f"min_CT={min(self.completion_times[-10:]):.2f}, "
                      f"PLoss={pol_loss:.4f}, VLoss={val_loss:.4f}")

        print("训练完成，总耗时：", time.time() - start)
        return self.episode_rewards, self.completion_times, best_schedule_log

#########################################################
# 主程序
#########################################################
def main(episode=None):

    env = AJSP_Environment()
    env.reset()

    # 获取动作特征维度
    ca = env.get_candidate_actions()
    caf = env.action_features(ca[0])
    feature_dim = len(caf[ca[0]])

    # 获取 CNN 状态维度
    cnn = CNNFeatureExtractor(input_channels=1).to(device)
    s = env.current_state
    feats = []
    for m in s:
        t = torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)
        feats.append(cnn(t))
    sf = torch.cat(feats, dim=1)
    state_dim = sf.shape[1]

    ppo = PPO(
        action_feature_dim=feature_dim,
        combined_dim=state_dim + feature_dim,
        mlp_value_dim=state_dim
    )

    trainer = Trainer(env, ppo, max_episodes=episode)
    episode_rewards, completion_times, best_schedule_log = trainer.train()

    torch.save({
        'policy': ppo.policy_net.state_dict(),
        'value': ppo.value_net.state_dict(),
        'reward': trainer.episode_rewards,
        'best_completion_time': min(completion_times),  # 添加最佳完工时间
        'best_schedule_log': best_schedule_log,  # 添加最佳调度日志
        'completion_times': completion_times,  # 添加所有完工时间
    }, "ppo_fajsp_model.pth")

    print("模型已保存 ppo_fajsp_model.pth")
    print(f"最佳完工时间: {min(completion_times)}")
    # =============================
    # 训练结束后绘制训练曲线（奖励 / 完工时间 / 损失 / 甘特图）
    # =============================
    rewards = trainer.episode_rewards
    completion_times = trainer.completion_times
    policy_losses = trainer.policy_losses
    value_losses = trainer.value_losses

    # --- 移动平均函数 ---
    def moving_average(data, window=10):
        ma = []
        for i in range(len(data)):
            if i < window:
                ma.append(np.mean(data[:i+1]))
            else:
                ma.append(np.mean(data[i-window:i]))
        return ma

    rewards_ma = moving_average(rewards)
    ct_ma = moving_average(completion_times)

    # --- 绘图 ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # ------------ (1) 奖励曲线 ------------
    ax = axs[0, 0]
    ax.plot(rewards, label='Reward', alpha=0.6)
    ax.plot(rewards_ma, label='Moving Avg (10)', linewidth=2)
    ax.set_title("Reward Curve", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)

    # ------------ (2) 完工时间曲线 ------------
    ax = axs[0, 1]
    ax.plot(completion_times, label='Completion Time', alpha=0.6)
    ax.plot(ct_ma, label='Moving Avg (10)', linewidth=2)
    ax.set_title("Completion Time Curve", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Completion Time")
    ax.legend()
    ax.grid(True)

    # ------------ (3) 策略损失（Policy Loss） ------------
    ax = axs[1, 0]
    ax.plot(policy_losses, label='Policy Loss', color='tab:orange')
    ax.set_title("Policy Loss", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # ------------ (4) 价值损失（Value Loss） ------------
    ax = axs[1, 1]
    ax.plot(value_losses, label='Value Loss', color='tab:green')
    ax.set_title("Value Loss", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # --- 布局与保存 ---
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    plt.show()
    # 绘制最优甘特图
    plot_best_gantt(
        best_schedule_log,
        kind_task_tuple=env.kind_task_tuple,
        machine_tuple=env.machine_tuple,
        filename="best_gantt.png"
    )

#########################################################
# 绘制甘特图
#########################################################
def plot_best_gantt(best_schedule_log, kind_task_tuple, machine_tuple, filename="best_gantt_chart.png"):
    """
    根据 best_schedule_log 绘制最优甘特图
    best_schedule_log 每条记录为：
        {
            "job": v,
            "op": j,
            "machine": m,
            "start": t_begin,
            "end":   t_end
        }
    """
    if not best_schedule_log:
        print("best_schedule_log 为空，无法绘制甘特图！")
        return False

    # ==========================================================
    #  设置字体与样式
    # ==========================================================
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150

    # 计算总完工时间
    completion_time = max([entry["end"] for entry in best_schedule_log])

    # 画布
    fig, ax = plt.subplots(figsize=(18, 10))

    # ==========================================================
    #  颜色映射：按 (v,j) 上色
    # ==========================================================
    colors = plt.cm.tab20(np.linspace(0, 1, len(kind_task_tuple)))
    color_dict = {task: colors[i] for i, task in enumerate(kind_task_tuple)}

    # ==========================================================
    #  y 轴机器坐标
    # ==========================================================
    machine_y_positions = {m:i for i,m in enumerate(sorted(machine_tuple))}

    # 机器时间线
    for m in sorted(machine_tuple):
        y = machine_y_positions[m]
        ax.hlines(y=y, xmin=0, xmax=completion_time, color="gray", alpha=0.4)

        # ax.text(-8, y, f"M{m}",
        #         ha='right', va='center', fontsize=12, fontweight='bold',
        #         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    # ==========================================================
    #  绘制甘特条
    # ==========================================================
    for entry_idx, entry in enumerate(sorted(best_schedule_log, key=lambda x: (x["machine"], x["start"]))):

        n = entry["number"]
        v = entry["job"]
        j = entry["op"]
        m = entry["machine"]
        t0 = entry["start"]
        t1 = entry["end"]
        dur = t1 - t0

        y = machine_y_positions[m]
        task_key = (v, j)
        # task_label = f"T{v}-{j}-{n}" #考虑批量
        task_label = f"T{v}-{j}"
        # 矩形块
        rect = patches.Rectangle(
            (t0, y - 0.35), dur, 0.7,
            linewidth=1.2,
            edgecolor="black",
            facecolor=color_dict[task_key],
            alpha=0.9
        )
        ax.add_patch(rect)

        # 中心文字
        ax.text(t0 + dur/2, y, task_label,
                ha="center", va="center",
                fontsize=14, fontweight="bold")
                # bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # # # 工序编号（序号）
        # ax.text(t0 + dur/2, y + 0.5, f"/{t0,t1}",
        #         ha="center", va="bottom",
        #         fontsize=17, color="red", fontweight="bold")

    # ==========================================================
    #  坐标轴设置
    # ==========================================================
    ax.set_title(
        f"最优调度甘特图\n完工时间: {completion_time:.2f}",
        fontsize=18, fontweight='bold'
    )
    ax.set_xlabel("时间", fontsize=14)
    ax.set_ylabel("机器", fontsize=14)

    ax.set_xlim(0, completion_time * 1.03)
    ax.set_ylim(-0.5, len(machine_tuple)-0.5)

    ax.set_yticks([machine_y_positions[m] for m in sorted(machine_tuple)])
    ax.set_yticklabels([f"M{m}" for m in sorted(machine_tuple)], fontsize=12)

    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # 图例
    legend_elements = [
        patches.Patch(facecolor=color_dict[task], label=f'T{task[0]}-{task[1]}')
        for task in kind_task_tuple
    ]
    ax.legend(handles=legend_elements, title="工序类型",
              fontsize=11, title_fontsize=12,
              bbox_to_anchor=(1.15, 1), loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"最优甘特图已保存为：{filename}")
    return True


if __name__ == '__main__':
    main(episode=500)
    # data_list = FT10C
    # fluid_ajsp = AJSP(data_list=data_list)
    # 打印所有属性
    # data = Data(data_list)
    # print("kind_task_tuple:", data.kind_task_tuple)
    # print("machine_count:", data.machine_count)
    # print("machine_tuple:", data.machine_tuple)
    # print("task_p_dict:", data.task_p_dict)
    # print("machine_pj_dict:", data.machine_pj_dict)
    # print("time_pjm_dict:", data.time_pjm_dict)
    # print("time_mpj_dict:", data.time_mpj_dict)
    # print("assemble_task:", data.assemble_task)
    # print("cost_pj_dict:", data.cost_pj_dict)
    # print("pre_pj_dict:", data.pre_pj_dict)
    # print("total_cost_dict:", data.total_cost_dict)
    # print("cost_count:", data.cost_count)
    # print("post_pj_dict:", data.post_pj_dict)
