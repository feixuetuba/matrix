#-*- coding:utf-8 -*-
import copy

def get_number_ops(num, op):
    if op == 'add':
        return type(num).__add__
    elif op == "sub":
        return type(num).__sub__
    elif op == 'mul':
        return type(num).__mul__
    elif op == 'div':
        return type(num).__div__  


class Vector:
    def __init__(self, data):
        self.data = data
    
    def __add__(self, vb):
        data = []
        for x, y in zip(self.data, vb.data):
            data.append(x + y)
        return Vector(data)

    def __sub__(self, vb):
        data = []
        for x, y in zip(self.data, vb.data):
            data.append(x - y)
        return Vector(data)
    
    def __mul__(self, x):
        data = []
        for d in self.data:
            data.append(d * x)
        return Vector(data)
    
    def __str__(self):
        return self.data.__str__()    

class MSequence:
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape
    
    def __get(self, data):
        if type(data) != list:
            yield data
        else:
            for d in data:
                for v in self.__get(d):
                    yield v
                
    def get(self):
        for v in self.__get(self.data):
            yield v
        
class Matrix:
    def __init__(self, data):
        if type(data) != list:
            raise Exception('input is :%s not a list'%type(data))
        
        self.data = copy.deepcopy(data)
        self.size = 1
        self.shape = self.get_shape(data)
        
        if type(self.shape) is not tuple:
            self.shape = (1, self.shape)
            self.data = [data]
        for s in self.shape:
            self.size = self.size * s
            
        self.rol = self.shape[0]
        self.col = self.shape[1]
        self.T = None
        
        #if init:
            #self.T = self.transpose()
            #self.I = self.inverse()
    
    def transpose(self, y=None):
        if self.T != None:
            return self.T
        data = []
        rols = self.shape[0]
        cols = self.shape[1]
        for col in xrange(cols):
            p = []
            for rol in xrange(rols):
                p.append(self.data[rol][col])
            data.append(p)    
        m = Matrix(data)
        self.T = m
        m.T = self
        return m

    def reshape(self, shape):
        def __fill_data(shape, index, ms):
            data = []
            if index == len(shape) - 1:
                for x,v in zip(xrange(shape[index]), ms):
                    data.append(v)
                return data
            else:
                for x in xrange(shape[index]):
                    data.append(__fill_data(shape, index+1, ms))
            return data
        s = 1
        _shape = []
        j = -1
        for i, x in enumerate(shape):
            s = s * x
            if x < 0:
                j = i
            _shape.append(x)
        
        if s < 0:
            if 1.0 * self.size % s  != 0.:
                raise Exception("can not reshape %r to %r"%(self.shape, shape))
        elif s > 0:
            if self.size != s:
                raise Exception("can not reshape %r to %r"%(self.shape, shape))
        else:
            raise Exception("can not reshape %r to %r"%(self.shape, shape))
        if j >= 0 :
            _shape[j] = self.size / (-s)
        shape  = tuple(_shape)
        ms = MSequence(self.data, self.shape).get()
        self.data = __fill_data(shape, 0, ms)
        self.shape = shape
        self.rol = self.shape[0]
        self.col = self.shape[1]
        return self

    def get_shape(self, d):
        if type(d[0]) != list:
            return len(d)
        else:
            return len(d), self.get_shape(d[0])
                
    def __add__(self, mb):
        if isinstance(mb, Matrix):
            return self.add_sub_matrix(mb, "add", 'l')
        else:
            return self.op_per_item(mb, "add", 'l')

    def __radd__(self, mb):
        if isinstance(mb, Matrix):
            return self.add_sub_matrix(mb, "add", 'r')
        else:
            return self.op_per_item(mb, "add", 'r')
    
    def __sub__(self, mb):
        if isinstance(mb, Matrix):
            return self.add_sub_matrix(mb, "sub", 'l')
        else:
            return self.op_per_item(mb, "sub", 'l')
    
    def __rsub__(self, mb):
        if isinstance(mb, Matrix):
            return self.add_sub_matrix(mb, "sub", 'r')
        else:
            return self.op_per_item(mb, "sub", 'r')
    
    def __mul__(self, mb):
        return self.op_per_item(mb, "mul")
    
    def __rmul__(self, mb):
        return self.op_per_item(mb, "mul")
    
    def __div__(self, mb):
        return self.op_per_item(mb, "div")
    
    def __getitem__(self, index):
        if type(index) == slice:
            return self.__getitems(self.data, [index], 0)            
        
        if type(index) is tuple:
            return self.__getitems(self.data, index, 0)
                
        elif type(index) is int:
            if index < 0 : index = 0
            if index > self.rol: index = self.rol
            return Matrix(self.data[index])

    
    def __setitem__(self, index, value):
        #only for two dim
        data = []
        if type(index) is int:
            data = self.data[index]
        
        if type(index) == tuple:
            #fix me latter, OH OH OH
            cur = index[1]
            if type(cur) is slice:
                startp = cur.start if cur.start != None else 0
                startp = max(startp, 0)
                endp = cur.stop if cur.stop != None else dlen
                endp = min(endp, dlen)
                data = self.data[index[0]][startp:endp] = value
            else:
                col = int(index[1])
                rol = int(index[0])
                data = self.data[rol][col] = value
            
        
    def __getitems(self, data, index, i):
            cur = index[i]
            dlen = len(data)
            if i == len(index) - 1:
                if type(cur) is int:
                    if cur < 0 :
                        cur += dlen
                    return data[cur]
                else:
                    d = []
                    startp = cur.start if cur.start != None else 0
                    if startp < 0 :
                        startp += dlen
                    endp = cur.stop if cur.stop != None else dlen
                    if endp < 0 :
                        endp += dlen
                    startp = max(startp, 0)
                    endp = min(endp, dlen)
                    for x in xrange(startp, endp):
                        d.append(data[x])
                    return Matrix(d)
            else:
                if type(cur) is int:
                    if cur < 0:
                        cur += dlen
                    return self.__getitems(data[cur], index, i+1)
                else:
                    d = []
                    startp = cur.start if cur.start != None else 0
                    if startp < 0 :
                        startp += dlen
                    endp = cur.stop if cur.stop != None else dlen
                    if endp < 0:
                        endp += dlen
                    startp = max(startp, 0)
                    endp = min(endp, dlen)
                    #print "***",startp, endp, len(data)
                    for x in xrange(startp, endp):
                        _d = self.__getitems(data[x], index, i+1)
                        if isinstance(_d, Matrix):
                            d.append(_d.data[0])
                        else:
                            d.append(_d)
                    return Matrix(d)
    
    def op_per_item(self, num, op="add", direct='l'):
        data = []
        func = get_number_ops(num, op)
        if direct == 'l':
            for r in xrange(self.rol):
                p = []
                for c in xrange(self.col):
                    p.append(func(self.data[r][c], num))
                data.append(p)
        else:
            for r in xrange(self.rol):
                p = []
                for c in xrange(self.col):
                    p.append(func(num, self.data[r][c]))
                data.append(p)
        return Matrix(data)
    
    def add_sub_matrix(self, m, op, direct='l'):
        if m.rol == 1 and m.col == 1:
            return self.op_per_item(m.data[0][0], op, direct)
        
        num = self.data[0][0]
        func = get_number_ops(num, op)  
        data = []
        rcount = 0
        ccount = 0
        left = None
        right = None
        if direct == 'l':
            left = self
            right = m
        else:
            right = self
            left = m
        
        if left.rol == right.rol and left.col == right.col:
            data = [[0 for x in xrange(self.col)] for y in xrange(self.rol)]
            for rol in xrange(m.rol):
                for col in xrange(m.col):
                    data[rol][col] = func(left.data[rol][col], right.data[rol][col])
        elif left.rol == right.rol:
            if left.col == 1:
                data = [[0 for x in xrange(right.col)] for y in xrange(right.rol)]
                for col in xrange(right.col):
                    for rol in xrange(right.rol):
                        data[rol][col] = func(left.data[rol][0],  right.data[rol][col])
            elif right.col == 1:
                data = [[0 for x in xrange(left.col)] for y in xrange(left.rol)]
                for col in xrange(left.col):
                    for rol in xrange(left.rol):
                        data[rol][col] = func(left.data[rol][col],  right.data[rol][0])
            else:
                raise Exception("operands could not be broadcast together with shapes (%d, %d) (%d, %d)"%(self.rol, self.col, m.rol, m.col))

        elif left.col == right.col:
           
            if left.rol == 1:
                data = [[0 for x in xrange(right.col)] for y in xrange(right.rol)]
                for rol in xrange(right.rol):
                    for col in xrange(right.col):
                        data[rol][col] = func(left.data[0][col], right.data[rol][col])
            elif right.rol == 1:
                data = [[0 for x in xrange(left.col)] for y in xrange(left.rol)]
                for rol in xrange(left.rol):
                    for col in xrange(left.col):
                        data[rol][col] = func(left.data[rol][col], right.data[0][col])
            else:
                raise Exception("operands could not be broadcast together with shapes (%d, %d) (%d, %d)"%(self.rol, self.col, m.rol, m.col))

        else:    
            raise Exception("operands could not be broadcast together with shapes (%d, %d) (%d, %d)"%(self.rol, self.col, m.rol, m.col))
            
        return Matrix(data)    
  
    def tolist(self):
        return self.data
        
    def __str__(self):
        return self.tolist().__str__()

    def __type__(self):
        return "Matrix"
        
def eye(rol, col):
    data = []
    for r in xrange(rol):
        if r >= col:
            break
        p = [0. for x in range(col)]
        p[r] = 1.
        data.append(p)
    return Matrix(data).reshape((rol, col))

def zeros(rol, col):
    data = [0 for x in range(rol * col)]
    return Matrix(data).reshape((rol, col))
    
def dot(ma, mb):
    data = []
    mb.transpose()
    if ma.col != mb.rol :
        raise Exception("can not dot (%d, %d) with (%d, %d)"%(ma.rol, ma.col, mb.rol, mb.col))
    
    for i in xrange(ma.rol):
        p = []
        for j in xrange(mb.T.rol):
            count = 0
            for x, y in zip(ma.data[i], mb.T.data[j]):
                count += x * y
            p.append(count)
        data.append(p)        
    return Matrix(data)

def sum(x):
    count = 0
    d = x
    if not isinstance(x, Matrix):
        d = Matrix(d)
    (rol, col) = d.shape
    count = 0
    for r in xrange(rol):
        for c in xrange(col):
            count += d.data[r][c]
    return count
    
def transpose(m):
    return m.transpose()

def reshape(m, shape):
    return m.reshape(shape)

def cast(m, dtype):
    for rol in xrange(m.rol):
        for col in xrange(m.col):
            m.data[rol][col] = dtype(m.data[rol][col])
    return m

#XW should be MxM, and Y should be Mx1
#if local is True, will use the local data of XW, Y
#while the XW is singular, an ignore is False, will raise Exception
def gauss(XW, Y, local=False, ignore=False):
    def exchange(a, b, srol, drol):
        t = a[srol]
        a[srol] = a[drol]
        a[drol] = t
        
        t = b[srol]
        b[srol] = b[drol]
        b[drol] = t
        
    rols, cols = XW.shape
    yr, yc = Y.shape
    if yr != rols or yc != 1:
        raise Exception('XW should be MxM, and Y should be Mx1')
    if not local:
        xw = copy.deepcopy(XW.data)
        y = copy.deepcopy(Y.data)
    else:
        xw = ma.data
        y = mb.data[0]
    #print "**********************"
    #print xw
    for i in xrange(0,rols - 1):
        if xw[i][i] == 0:
            for j in xrange(i+1, rols):
                if xw[j][i] != 0:
                    exchange(xw, y, i, j)
                    break
                if j == rols - 1:
                    if not ignore:
                        raise Exception("Matrix XW is singular!")
                    else:
                        xw[i][i] = 1000000000000
        for j in range(i+1, rols):
            if xw[j][i] != 0.0:
                lam = float(xw[j][i])/xw[i][i]
                for p in range(i+1, rols):
                    xw[j][p] = xw[j][p] - lam * xw[i][p]
                y[j][0] = y[j][0] - lam * y[i][0]
    if xw[i+1][i+1] == 0:
        xw[i+1][i+1] = 1000000000000
    #print xw
    #print "************************"
    result = []
    _rols = rols-1
    for k in range(_rols, -1, -1):
        count = 0
        for j in xrange(_rols, k, -1):
            count += xw[k][j] * result[_rols - j]
        #if xw[k][k] == 0:
            #print (rols, rols,(i,i))
        result.append((y[k][0] - count) / xw[k][k])
    result.reverse()
    return Matrix(result).transpose()
 
if __name__ == '__main__':

    v1 = Vector([1,2,3])
    v2 = v1 * 5
    print v1
    print v2
    print v1 - v2
    print v1 + v2

    ma = Matrix(range(25))
    ma.reshape((5, 5))
    mb = Matrix([1,2,3,4,5])
    print "ma",ma
    print "ma.shape", ma.shape
    print "ma.T", transpose(ma)
    print "ma.T.shape",ma.T.shape
    print "ma[1]",ma[1]
    print "ma[0:2]",ma[0:2]
    print "ma[:, :2]",ma[:, :2]
    print "ma[0:2,0:2]",ma[0:2, 0:2]
    print "ma[0:-2,0:2]",ma[0:-2, 0:2]
    print "ma[0:-2,0:-2]",ma[0:-2, 0:-2]
    print "ma X ma",dot(ma, ma)
    print "ma - 5", ma - 5
    print "ma - zeros(5,5)", ma - cast(zeros(5,5), int)
    print "ma * 5", ma * 5
    print "ma / 5", ma / 5
    print "ma[1,1] - ma[0,1]",ma[1,1] - ma[0,1]
    print "ma[1,1] + ma[0,1]",ma[1,1] + ma[0,1]

    transpose(mb)
    print "mb", mb
    print "mb.T", mb.T
    print "ma + mb", ma + mb
    print "mb + ma", mb + ma
    print "ma + mb.T", ma + mb.T
    print "mb.T + ma", mb.T + ma
    
    print "eye",eye(3,3)
    
    print "Gausss"
    x = Matrix([[2,1,-1,1],[-3,-1,2,1],[-2,1,2,1],[1,1,1,1]])
    y = reshape(Matrix([9,-10,-2,5]),(4,1))
    print "dot(X, W):",x
    print "Y:",y
    result = gauss(x,y)
    print "resut(X):",result
    print "result.T:",result.T 
    exit()
    print "******"
    '''
    ma = Matrix(range(9), (3 , 3))
    print "eye",eye(3,3)
    print "ma",ma
    print "ma.T", transpose(ma)
    print "ma.T",ma.T
    print "ma.T.T.T.T",ma.T.T.T.T
    print "ma X maa",dot(ma, ma)
    print "ma X ma.T",dot(ma, ma.T)
    print "ma[1]",ma[1]
    print "ma[0:2]",ma[0:2]
    print "ma[0:2,0:2]",ma[0:2, 0:2]
    print "ma - 5", ma - 5
    print "type(ma[1,1])",type(ma[1,1])
    print "ma[1,1], ma[0, 1]", ma[1,1],  ma[0,1]
    print "ma[1,1] - ma[0,1]",ma[1,1] - ma[0,1]
    print "ma[1,1] + ma[0,1]",ma[1,1] + ma[0,1]
    
    mb = Matrix([1,2,3], 1, 3)
    transpose(mb)
    print "mb", mb
    print "mb.T", mb.T
    print "ma + mb", ma + mb
    print "mb + ma", mb + ma
    print "ma + mb.T", ma + mb.T
    print "mb.T + ma", mb.T + ma
    print"******"
    for m in ma:
        print m
    for m1, m2 in zip(ma, ma.T):
        print m1,m2
    print "m1.shape",m1[0,0].shape
    
    mb = Matrix(range(3), (1, 3))
    print dot(ma, mb)
    '''