import pandas

#df = pandas.DataFrame({'filename' :  ['test0.dat', 'test2.dat'], 'cat': ['o', 'v']})
# df = pandas.read_csv("test.csv")
# df2 = pandas.DataFrame({'filename' :  'test2.dat', 'cat':'q'}, index=[0])
#
# print("1",df)
#
# df.set_index('filename', inplace=True)
# df2.set_index('filename', inplace=True)
# df.update(df2)
#
# print("2",df)
#
# df2 = pandas.DataFrame({'filename' :  'test1.dat', 'cat':'b'}, index=[0])
#
# df.set_index('filename', inplace=True)
# df2.set_index('filename', inplace=True)
# df.update(df2)
#
# print("3",df)
#
# df.to_csv("test.csv")

# df = pandas.read_csv("test.csv")
df = pandas.DataFrame({'filename' :  ['test0.dat', 'test2.dat'], 'cat': ['o', 'v']})
df2 = pandas.DataFrame({'filename' :  ['test2.dat'], 'cat':['q']})
df.set_index('filename',inplace=True)

print("AAAA",df)

#df['cat'] = df2.set_index(['filename'])['cat'].combine_first(df.set_index(['filename'])['cat']).values
df2.set_index('filename',inplace=True)
#df.update(df2)

df = pandas.concat([df[~df.index.isin(df2.index)], df2])

print("BBBB",df)

df2 = pandas.DataFrame({'filename' :  ['test0.dat'], 'cat':['b']})
df2.set_index('filename',inplace=True)

#df['cat'] = df2.set_index(['filename'])['cat'].combine_first(df.set_index(['filename'])['cat']).values
#df.update(df2)
df = pandas.concat([df[~df.index.isin(df2.index)], df2])

print("CCCC",df)

df.to_csv("test.csv")
