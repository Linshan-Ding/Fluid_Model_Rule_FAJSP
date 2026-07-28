# -*- coding: utf-8 -*-
"""
D1-D12 分布偏移(OOD)算例配置。

所有算例基于C13-C15的中规模配置(M_p=8, M_a=4, P=4, R=8, J_r=4),
每组只改动一个生成因子, 其余保持训练分布(与论文6.10节一致):
  Shift-PT    (D1-D3) : 加工时间 U[10,50] (训练分布为U[10,20])
  Shift-alpha (D4-D6) : 装配消耗系数 U[1,5] (训练分布为U[1,2])
  Shift-BOM   (D7-D9) : 三层BOM(组件->2个子装配体->成品)
  Shift-Elig  (D10-D12): 机器可选模式 sparse(每工序1-2台)/dense(同类全部)
"""

BASE = dict(M_p=8, M_a=4, product_count=4, kind_count=8, J_r=4)

OOD_CONFIGS = {
    'D1':  dict(shift='Shift-PT',    env_kwargs=dict(**BASE, N_p=2, pt_range=(10, 50), instance_name='D1_pt_n2')),
    'D2':  dict(shift='Shift-PT',    env_kwargs=dict(**BASE, N_p=4, pt_range=(10, 50), instance_name='D2_pt_n4')),
    'D3':  dict(shift='Shift-PT',    env_kwargs=dict(**BASE, N_p=6, pt_range=(10, 50), instance_name='D3_pt_n6')),
    'D4':  dict(shift='Shift-alpha', env_kwargs=dict(**BASE, N_p=2, alpha_range=(1, 5), instance_name='D4_alpha_n2')),
    'D5':  dict(shift='Shift-alpha', env_kwargs=dict(**BASE, N_p=4, alpha_range=(1, 5), instance_name='D5_alpha_n4')),
    'D6':  dict(shift='Shift-alpha', env_kwargs=dict(**BASE, N_p=6, alpha_range=(1, 5), instance_name='D6_alpha_n6')),
    'D7':  dict(shift='Shift-BOM',   env_kwargs=dict(**BASE, N_p=2, bom_mode='three_level', instance_name='D7_bom_n2')),
    'D8':  dict(shift='Shift-BOM',   env_kwargs=dict(**BASE, N_p=4, bom_mode='three_level', instance_name='D8_bom_n4')),
    'D9':  dict(shift='Shift-BOM',   env_kwargs=dict(**BASE, N_p=6, bom_mode='three_level', instance_name='D9_bom_n6')),
    'D10': dict(shift='Shift-Elig(sparse)', env_kwargs=dict(**BASE, N_p=4, elig_mode='sparse', instance_name='D10_elig_sparse_n4')),
    'D11': dict(shift='Shift-Elig(sparse)', env_kwargs=dict(**BASE, N_p=6, elig_mode='sparse', instance_name='D11_elig_sparse_n6')),
    'D12': dict(shift='Shift-Elig(dense)',  env_kwargs=dict(**BASE, N_p=4, elig_mode='dense',  instance_name='D12_elig_dense_n4')),
}

ORDER = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12']
