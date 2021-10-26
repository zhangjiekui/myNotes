def fun(x,y=0):
    print("传入的参数",x,y)
    return x+y
for each in map(fun, [1,2],[3,4]):
    print("求和为：",each)
    print("----------")
print('========================================')
print(list(zip([1,2],[3,4])))
for each in map(fun, *zip([1,2],[3,4])):
    print("求和为：",each)
    print("----------")
