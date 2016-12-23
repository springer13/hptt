def genMacro(prefetchDistance, blockingA, blockingB ):
   blockingMicro = 8
   numMicroKernels = blockingA * blockingB
   for i in range(blockingA):
      for j in range(blockingB):
         block = 0
         iPrefetch = i
         jPrefetch = j
         prefetchTodo = prefetchDistance
         while prefetchTodo >0:
             jPrefetch = (jPrefetch + 1)%blockingB
             if( jPrefetch == 0 ):
                 iPrefetch += 1
                 if( iPrefetch == blockingA ):
                     iPrefetch = 0
                     block += 1
             prefetchTodo -= 1
             
         offsetA = "(%d * lda + %d)"%(jPrefetch*blockingMicro,iPrefetch*blockingMicro)
         offsetB = "(%d * ldb + %d)"%(iPrefetch*blockingMicro,jPrefetch*blockingMicro)
         Aprefetch = "A"
         Bprefetch = "B"
         if( block > 0 ):
            Aprefetch = "Anext"
            Bprefetch = "Bnext"
         if( jPrefetch % 2 == 0 ):
            print "prefetch(%s + %s, ldb);"%(Bprefetch, offsetB)
         if( iPrefetch % 2 == 0 ):
            print "prefetch(%s + %s, lda);"%(Aprefetch, offsetA)
         offsetA = "(%d * lda + %d)"%(j*blockingMicro,i*blockingMicro)
         offsetB = "(%d * ldb + %d)"%(i*blockingMicro,j*blockingMicro)
         print "sTranspose8x8<betaIsZero>(A + %s, lda, B + %s, ldb  , reg_alpha , reg_beta);"%(offsetA, offsetB)

genMacro(8, 4, 2)
