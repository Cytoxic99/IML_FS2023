def test1(opos, x):
    res = opos+1
    def test2(x, res):
        res+=x
        return res

    return test2

print(test1(1, 1))


