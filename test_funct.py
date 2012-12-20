import pylab
import test_prob


a = [test_prob.gaussNoise((0,0,0),(0,0,0)) for _ in range(10000)]

ax1 = pylab.subplot("211")
ax1.hist(a)

ax2 = pylab.subplot("212")

a.sort()
ax2.plot(a)
pylab.show()
