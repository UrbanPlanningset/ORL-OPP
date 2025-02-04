# foo = ['a', 'b', 'c', 'd', 'e']
# from random import randrange
# random_index = randrange(0,len(foo))
# print(foo[random_index])
import json

import numpy as np

with open('E:/数据1/porto_only_waynode', mode='r', encoding='gbk') as f:
    porto_only_waynode = json.load(f)
print(len(porto_only_waynode))
base_state = [41.1414543, -8.6186966]
target_state = [41.154489, -8.630838]
min_jing_state=min(base_state[0],target_state[0])
max_jing_state=max(base_state[0],target_state[0])
min_wei_state=min(base_state[-1],target_state[-1])
max_wei_state=max(base_state[-1],target_state[-1])
'''经度：max=[41.1971976] min=[41.1260104]
   纬度：max=[-8.5472757] min=[-8.6949287,]'''
porto=[]
for k, v in porto_only_waynode.items():
    porto.append(v)

next_state=[]
for i in porto:
    if base_state[0]-0.00001 < i[0] < base_state[0]+0.00001 :
        next_state.append(i)

print(len(next_state))