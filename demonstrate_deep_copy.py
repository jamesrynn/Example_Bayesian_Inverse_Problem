d = dict()
d['apple'] = 0.80
d['banana'] = 0.4
d['melon'] = 1.20

for key in d:
    print(key, d[key])


# shallow copy (new compound object but references to original)
d1 = d
d1['apple'] = 'sold out'
print('shallow copy:', d1['apple'], d['apple'])

from copy import deepcopy
# deep copy
d2 = deepcopy(d)
d2['apple'] = 0.80
print('deep copy:', d2['apple'], d['apple'])
