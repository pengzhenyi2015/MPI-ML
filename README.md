# MPI-ML
Simple MPI Programs，and Meachine Learning Programs written by MPI-C

# KNN
A KNN algorithm （Euclidean, L2） to recongnize handwriting digits from 0 to 9

Dataset：mnist http://yann.lecun.com/exdb/mnist/

How to use: mpicc knn.c -o knn -O -Wall

mpirun --allow-run-as-root -np 4 -host gpu1 knn --trainset data_training --testset data_test --trainsize 60000 --testsize 10000

Outputs:

Thu Feb 25 14:24:43 2016[1,0]<stdout>:Test 1000 samples, correct rate:0.854000

Thu Feb 25 14:24:55 2016[1,0]<stdout>:Test 2000 samples, correct rate:0.851500

Thu Feb 25 14:25:08 2016[1,0]<stdout>:Test 3000 samples, correct rate:0.856667

Thu Feb 25 14:25:20 2016[1,0]<stdout>:Test 4000 samples, correct rate:0.856500

Thu Feb 25 14:25:33 2016[1,0]<stdout>:Test 5000 samples, correct rate:0.859000

Thu Feb 25 14:25:45 2016[1,0]<stdout>:Test 6000 samples, correct rate:0.863667

Thu Feb 25 14:25:57 2016[1,0]<stdout>:Test 7000 samples, correct rate:0.869571

Thu Feb 25 14:26:10 2016[1,0]<stdout>:Test 8000 samples, correct rate:0.876875

Thu Feb 25 14:26:22 2016[1,0]<stdout>:Test 9000 samples, correct rate:0.882556

Thu Feb 25 14:26:34 2016[1,0]<stdout>:Test 10000 samples, correct rate:0.883200

Well, the error rate is pretty high,~~~~(>_<)~~~~
