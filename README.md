# MPI-ML
Simple MPI Programs，and Meachine Learning Programs written by MPI-C

# KNN
A KNN algorithm （Euclidean, L2） to recongnize handwriting digits from 0 to 9

Dataset：mnist http://yann.lecun.com/exdb/mnist/

How to use: mpicc knn.c -o knn -O -Wall

mpirun --allow-run-as-root -np 4 -host gpu1 knn --trainset data_training --testset data_test --trainsize 60000 --testsize 10000

Outputs （Euclidean L2）:

Test 1000 samples, correct rate:0.854000

Test 2000 samples, correct rate:0.851500

Test 3000 samples, correct rate:0.856667

Test 4000 samples, correct rate:0.856500

Test 5000 samples, correct rate:0.859000

Test 6000 samples, correct rate:0.863667

Test 7000 samples, correct rate:0.869571

Test 8000 samples, correct rate:0.876875

Test 9000 samples, correct rate:0.882556

Test 10000 samples, correct rate:0.883200

Well, the error rate is pretty high,~~~~(>_<)~~~~
