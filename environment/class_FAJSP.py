"""
柔性装配作业车间调度
"""

from docplex.mp.model import Model
from Instance_generate import Instance

class Base():
    def __init__(self, count_kind):
        self.count_kind = count_kind  # 各类工件的数量

class Kind():
    """
    工件和装配体类型类
    装配体类型类
    凑齐装配所需一组组件视为一个装配体,以[组件1,组件2,...]的形式表示中
    需要构建一个方法用于构建装配体
    """

    def __init__(self, p):
        self.kind = p #工件类型 + 产品类型
        self.job_arrive_list = []  # 已经到达的工件对象列表
        self.job_unprocessed_list = []  # 未加工完成的工件对象列表

    """该类型部件已到达工件数:下一阶段的部件n起始编号"""
    @property
    def number_start(self):
        return len(self.job_arrive_list)



class Tasks(Kind):
    """定义工序o_pj类"""
    def __init__(self, p , j):
        Kind.__init__(self, p)  #调用父类的构造函数
        # 基本属性
        self.task = j  # 所属工序
        self.machine_tuple = None  # 可选加工机器编号
        self.time_remain = None  # 后续剩余处理时间
        # 附加属性
        self.job_now_list = []  # 处于该工序段的工件对象列表
        self.job_unprocessed_list = []  # 该工序段未被加工的工件对象列表
        self.task_unprocessed_list = []  # 该工序段还未加工的工序对象列表
        self.task_processed_list = []  # 该工序段已加工的工序对象列表
        # 流体相关属性
        self.fluid_process_rate_m_dict = {}  # 流体中被各机器加工的速率
        self.fluid_machine_list = []  # 流体模型解可选加工机器
        self.fluid_time_sum = None  # 流体模型中该工序的加工时间
        self.fluid_rate_sum = None  # 流体模型中加工该工序的速率
        self.fluid_number = None  # 处于该工序段的流体数量
        self.fluid_unprocessed_number = None  # 未被加工的流体数
        self.fluid_unprocessed_number_start = None  # 订单到达时刻未被加工的流体数

class Job(Kind):
    """部件类"""
    def __init__(self, p ,n):
        Kind.__init__(self, p)  # 调用父类的构函
        # 基本属性
        self.number = n  # 该部件的第几个工件
        self.time_arrive = None  # 该部件的到达时间
        # 附加属性
        self.task_list = []  # 分配机器的工序对象列表
        self.task_unprocessed_list = []  # 未分配机器的工序对象列表

class Task(Tasks, Job):
    """工序类"""
    def __init__(self, p, n, j):
        Tasks.__init__(self, p, j)  # 调用父类的构函
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
        self.task_list = []  # 机器已加工工序对象列表
        self.job_object = None  # 机器正在处理的工件对象
        self.unprocessed_rj_dict = {}  # 未被m加工的各工序类型的工序总数/随着加工过程动态更新
        # 流体附加属性
        self.fluid_kind_task_list = []  # 可选加工工序类型
        self.time_ratio_rj_dict = {}  # 流体解中分配给各工序类型的时间比例
        self.fluid_process_rate_rj_dict = {}  # 流体解中加工各工序类型的速率
        self.fluid_unprocessed_rj_dict = {}  # 未被机器m加工的各工序类型流体总数
        self.fluid_unprocessed_rj_arrival_dict = {}  # 订单到达时刻未被m加工的各工序类型流体数

class FAJSP(Instance):
    """柔性装配作业车间调度类"""
    def __init__(self, M_p, M_a, N):
        Instance.__init__(self, M_p, M_a, N)  # 调用父类的构函
        # 实例化工件类型、工件、工序类型、工序和机器对象字典
        self.kind_task_dict = {(p, j): Tasks(p, j) for p in self.kind_tuple for j in self.task_p_dict[p]}  # 工序类型对象字典
        self.kind_dict = {p: Kind(p) for p in self.kind_tuple}  # 工件类型对象字典
        self.machine_dict = {m: Machine(m) for m in self.machine_tuple}  # 机器对象字典
        self.task_dict = {}  # (r,n,j) 工序对象字典 订单到达更新
        self.job_dict = {}  # (r,n)  # 工件对象字典
        self.process_rate_m_rj_dict = {m: {(p, j): 1 / self.time_mpj_dict[m][(p, j)] for (p, j) in self.kind_task_m_dict[m]} for m in self.machine_tuple}  # 机器加工流体速率
        self.reset_parameter()
        self.reset_object()
        print("成功定义FAJSP类")

    def reset_parameter(self):
        """初始化各字典和参数"""
        for p, kind in self.kind_dict.items():
            kind.job_arrive_list = []  # 已经到达的工件对象列表
            kind.job_unprocessed_list = []  # 未分配机器的工件对象列表
        for (p, j), kind_task_object in self.kind_task_dict.items():
            kind_task_object.machine_tuple = self.machine_pj_dict[(p, j)]  # 可选加工机器编号元组
            kind_task_object.time_remain = sum(self.time_pj_dict[(p, j)] for j in self.task_p_dict[p][j:])  # 后续剩余处理时间
            kind_task_object.job_now_list = []  # 处于该工序段的工件对象列表
            kind_task_object.job_unprocessed_list = []  # 该工序段未被加工的工件对象列表
            kind_task_object.task_unprocessed_list = []  # 该工序段还未加工的工序对象列表
            kind_task_object.task_processed_list = []  # 该工序段已加工的工序对象列表
        for m, machine_object in self.machine_dict.items():
            machine_object.kind_task_tuple = self.kind_task_m_dict[m]  # 可选加工工序类型元组
            machine_object.machine_state = 0  # 机器状态
            machine_object.time_end = 0  # 机器完工时间
            machine_object.task_list = []  # 机器已加工工序对象列表
            machine_object.job_object = None

    def reset_fluid_parameter(self):
            """初始化流体属性参数"""
            for (p, j), kind_task_object in self.kind_task_dict.items():
                kind_task_object.fluid_machine_list = []  # 流体模型中可选加工机器
                kind_task_object.fluid_process_rate_m_dict = {}  # 被各机器加工的速率
            for m, machine_object in self.machine_dict.items():
                machine_object.fluid_kind_task_list = []  # 流体解中可选加工工序类型列表
                machine_object.time_ratio_rj_dict = {}  # 流体解中分配给各工序类型的时间比例
                machine_object.fluid_process_rate_rj_dict = {}  # 流体解中加工各工序类型的速率
                machine_object.unprocessed_rj_dict = {}  # 未被m加工的工序o_rj的总数 (r,j)
                machine_object.fluid_unprocessed_rj_dict = {}  # 流体解中未被机器m加工的各工序类型总数
                machine_object.fluid_unprocessed_rj_arrival_dict = {}  # 订单到达时刻未被m加工的各工序类型数量

    def reset_object(self):
            """
            添加工件对象和工序对象+更新流体模型和属性
            :return: 添加工序对象和机器对象+更新流体模型和属性
            """
            # 初始化工件类型字典、工序类型对象字典、工序对象字典、工件对象字典
            for p in self.kind_tuple:
                n_start = self.kind_dict[p].number_start #装配过程每次装配，装配体的数量是不同的，而我们的p是产品为索引
                n_end = n_start + self.kind_number[p]
                for n in range(n_start, n_end):
                    job_object = Job(p, n)  # 实例化工件对象
                    job_object.task_list = []
                    job_object.task_unprocessed_list = []

                    self.kind_dict[p].job_arrive_list.append(job_object)
                    self.kind_dict[p].job_unprocessed_list.append(job_object)
                    self.job_dict[(p, n)] = job_object  # 加入工件字典
                    self.kind_task_dict[(p, 0)].job_now_list.append(job_object)
                    for j in self.task_p_dict[p]:
                        task_object = Task(p, n, j) #工序对象
                        job_object.task_unprocessed_list.append(task_object)  # 加入工序未处理工序对象字典
                        self.kind_task_dict[(p, j)].job_unprocessed_list.append(job_object)
                        self.kind_task_dict[(p, j)].task_unprocessed_list.append(task_object)
                        self.task_dict[(p, n, j)] = task_object  # 加入工序字典
            # 初始化流体属性
            for (p, j), kind_task_object in self.kind_task_dict.items():
                kind_task_object.fluid_number = len(kind_task_object.job_now_list)  # 处于该工序段的流体数量
                kind_task_object.fluid_unprocessed_number = len(kind_task_object.task_unprocessed_list)  # 未被加工的流体数
                kind_task_object.fluid_unprocessed_number_start = len(
                    kind_task_object.task_unprocessed_list)  # 订单到达时刻未被加工的流体数量
            # 求解流体模型更新流体模型属性
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
        var_list = {(m, (p, j)) for m in self.machine_tuple for (p, j) in self.kind_task_m_dict[m]}
        X = model.continuous_var_dict(var_list, lb=0, ub=1, name='X')
        # 各流体初始未加工数量
        fluid_number = {(p, j): self.kind_task_dict[(p, j)].fluid_unprocessed_number_start
                        for (p, j) in self.kind_task_tuple}
        # 各流体初始瞬态数量
        fluid_number_time = {(p, j): self.kind_task_dict[(p, j)].fluid_number for (p, j) in self.kind_task_tuple}
        process_rate_pj_sum = {(p, j): sum(X[m, (p, j)] * self.process_rate_m_rj_dict[m][(p, j)]
                                           for m in self.machine_pj_dict[(p, j)]) for (p, j) in self.kind_task_tuple}
        # 定义目标函数
        model.maximize(model.min(process_rate_pj_sum[(p, j)] / fluid_number[(p, j)] for (p, j) in self.kind_task_tuple))
        # 添加约束条件
        # 机器利用率约束
        model.add_constraints(model.sum(X[m, (r, j)] for (r, j) in self.kind_task_m_dict[m]) <= 1
                              for m in self.machine_tuple)
        # 解的可行性约束
        """
        对于工件r的当前工序j，假设其前一道工序为j-1，则必须满足：
        流体在工序j-1上的加工速率 >= 流体在工序j上的加工速率
        但对于装配工序，当前工序的流体加工速率应小于等于其所有前置工序的流体加工速率
        统一两阶段的表示方法为：
        为所有工序建立前置工序集合pre_pj_set
        """
        #工件r的可行性约束 (P,J)表示当前工序， (p,j)表示前置工序
        for (P, J) in self.kind_task_tuple:
            if fluid_number_time[(P, J)] == 0 and self.pre_pj_dict[(P, J)]:
                for (p, j) in self.pre_pj_dict[(P, J)]:
                    model.add_constraint(process_rate_pj_sum[(p, j)] >= process_rate_pj_sum[(P, J)])
        # 求解模型
        solution = model.solve()
        x = solution.get_value_dict(X)
        # 输出流体完工时间
        process_rate_pj_sum = {(p, j): sum(x[m, (p, j)] * self.process_rate_m_rj_dict[m][(p, j)]
                                           for m in self.machine_pj_dict[(p, j)]) for (p, j) in self.kind_task_tuple}
        fluid_completed_time = max(
            fluid_number[(p, j)] / process_rate_pj_sum[(p, j)] for (p, j) in self.kind_task_tuple)
        print("流体完工时间：", fluid_completed_time)
        print(solution)
        return x

    def update_fluid_parameter(self, x):
        """基于流体解更新流体参数"""
        for (m, (p, j)), rate in x.items():
            machine_object = self.machine_dict[m]
            kind_task_object = self.kind_task_dict[(p, j)]
            machine_object.time_ratio_rj_dict[(p, j)] = rate  # 流体解中分配给各工序类型的时间比例
            kind_task_object.fluid_process_rate_m_dict[m] = rate*self.process_rate_m_rj_dict[m][(p, j)]
            machine_object.fluid_process_rate_rj_dict[(p, j)] = rate*self.process_rate_m_rj_dict[m][(p, j)]
            if rate != 0:
                machine_object.fluid_kind_task_list.append((p,j))
                kind_task_object.fluid_machine_list.append(m)
        for (p, j), kind_task_object in self.kind_task_dict.items():
            kind_task_object.fluid_rate_sum = sum(kind_task_object.fluid_process_rate_m_dict.values())  # 工序类型处理速率
            kind_task_object.fluid_time_sum = 1 / kind_task_object.fluid_rate_sum  # 工序类型的加工时间
        for m, machine_object in self.machine_dict.items():
            for (p, j) in machine_object.kind_task_tuple:  # 添加除流体模型中可选工序类型外的工序
                kind_task_object = self.kind_task_dict[(p, j)]
                # 订单到达时刻未被m加工的各工序类型数量
                machine_object.fluid_unprocessed_rj_arrival_dict[(p, j)] = \
                    kind_task_object.fluid_unprocessed_number_start*\
                    machine_object.fluid_process_rate_rj_dict[(p, j)]/kind_task_object.fluid_rate_sum
                # 未被m加工的工序o_rj的总数 (r,j)
                machine_object.unprocessed_rj_dict[(p, j)] = machine_object.fluid_unprocessed_rj_arrival_dict[(p, j)]
                # 流体解中未被机器m加工的各工序类型总数
                machine_object.fluid_unprocessed_rj_dict[(p, j)] = machine_object.fluid_unprocessed_rj_arrival_dict[(p, j)]


if __name__ == "__main__":
    fajsp_instance = FAJSP(10, 5, 0)


