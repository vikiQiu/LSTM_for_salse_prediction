__author__ = 'VikiQiu'

def getTestError(testX, testY, sess):
    nPre = len(testY)
    for i in range(nPre):
        pre = sess.run(prediction)