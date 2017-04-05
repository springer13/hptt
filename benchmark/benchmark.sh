rm -f ttc_benchmark.dat
for i in `seq 1 5`;
do
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 2 1 0 7264 7264 | grep GiB > tmp.dat 
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 2 1 0 43408 1216 | grep GiB >> tmp.dat 
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 2 1 0 1216 43408  | grep GiB >> tmp.dat

   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 0 2 1 368 384 384  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 0 2 1 2144 64 384  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 0 2 1 368 64 2307  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 1 0 2 384 384 355  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 1 0 2 2320 384 59  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 1 0 2 384 2320 59  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 2 1 0 384 355 384  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 2 1 0 2320 59 384  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 3 2 1 0 384 59 2320  | grep GiB >> tmp.dat
    
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 0 3 2 1 80 96 75 96  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 0 3 2 1 464 16 75 96  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 0 3 2 1 80 16 75 582  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 2 1 3 0 96 75 96 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 2 1 3 0 608 12 96 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 2 1 3 0 96 12 608 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 2 0 3 1 96 75 96 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 2 0 3 1 608 12 96 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 2 0 3 1 96 12 608 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 1 0 3 2 96 96 75 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 1 0 3 2 608 96 12 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 1 0 3 2 96 608 12 75  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 3 2 1 0 96 75 75 96  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 3 2 1 0 608 12 75 96  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 4 3 2 1 0 96 12 75 608  | grep GiB >> tmp.dat

   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 0 4 2 1 3 32 48 28 28 48  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 0 4 2 1 3 176 8 28 28 48  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 0 4 2 1 3 32 8 28 28 298  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 3 2 1 4 0 48 28 28 48 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 3 2 1 4 0 352 4 28 48 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 3 2 1 4 0 48 4 28 352 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 2 0 4 1 3 48 28 48 28 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 2 0 4 1 3 352 4 48 28 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 2 0 4 1 3 48 4 352 28 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 1 3 0 4 2 48 48 28 28 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 1 3 0 4 2 352 48 4 28 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 1 3 0 4 2 48 352 4 28 28  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 4 3 2 1 0 48 28 28 28 48  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 4 3 2 1 0 352 4 28 28 48  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 5 4 3 2 1 0 48 4 28 28 352  | grep GiB >> tmp.dat

   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 0 3 2 5 4 1 16 32 15 32 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 0 3 2 5 4 1 48 10 15 32 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 0 3 2 5 4 1 16 10 15 103 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 3 2 0 5 1 4 32 15 15 32 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 3 2 0 5 1 4 112 5 15 32 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 3 2 0 5 1 4 32 5 15 112 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 2 0 4 1 5 3 32 15 32 15 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 2 0 4 1 5 3 112 5 32 15 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 2 0 4 1 5 3 32 5 112 15 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 3 2 5 1 0 4 32 15 15 32 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 3 2 5 1 0 4 112 5 15 32 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 3 2 5 1 0 4 32 5 15 112 15 15  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 5 4 3 2 1 0 32 15 15 15 15 32  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 5 4 3 2 1 0 112 5 15 15 15 32  | grep GiB >> tmp.dat
   KMP_AFFINITY=compact,1 OMP_NUM_THREADS=24 ./benchmark.exe 6 5 4 3 2 1 0 32 5 15 15 15 112  | grep GiB >> tmp.dat
   python maxFromFiles.py tmp.dat hptt_benchmark.dat
done
