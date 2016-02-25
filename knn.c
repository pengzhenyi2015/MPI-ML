/*
 * Copyright: pengzhenyi
 * Author: pengzhenyi2015
 * Description: A KNN algorithm, Use mnist dataset to recongnize handwriting numbers
 * Usage: mpicc knn.c -o knn -O -Wall
 *        --trainset <trainset directory>
 *        --testset <testset directory>
 *        --trainsize <item count of trainset>
 *        --testsize <item count of testset>
 * */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>

#include <mpi.h>

#define FILENAME_LENGTH 100
#define DIMS 1000
#define FEATURE_NUM 784 //28 * 28

#define K_KNN 100
#define LABEL_NUM 10

#define POWER_OF_2(x) ((x) *(x))

//#define DEBUG

typedef struct {
    double label;
    double distance;
} knn_result;

int readfile_to_buffer(char* filename, double* dataset, int datasize);
int knn_check_results_reduce(knn_result* results, int neighbours);
int knn_get_neighbours_map(double* train_batch, int train_batch_size, double* test_record,
                           knn_result* results);


int main(int argc, char** argv) {
    char trainset_dir[FILENAME_LENGTH];
    char testset_dir[FILENAME_LENGTH];
    int train_size = 0;
    int test_size = 0;
    int test_size_read = 0;
    int train_size_read = 0;
    double* trainset = NULL;
    double* train_batch = NULL;
    double* test_record = NULL;
    int* send_displs = NULL;
    int* send_counts = NULL;
    double* testset = NULL;
    int i = 0;
    int j = 0;

    int comm_rank = -1;
    int comm_size = 0;
    MPI_Datatype knn_result_type;

    int opt = 0;
    int option_index = 0;
    static struct option long_options[] = {
        {"trainset", 1, 0, 'n'},
        {"testset", 1, 0, 't'},
        {"trainsize", 1, 0, 's'},
        {"testsize", 1, 0, 'i'},
        {0, 0, 0, 0}
    };

    if (argc < 8) {
        printf("Usage: knn --trainset <trainset directory> --testset <testset directory>\
            --trainsize <trainsize rows> --testsize <testsize rows>\n");
        return 0;
    }

    while ((opt = getopt_long(argc, argv, "i:o:", long_options, &option_index)) != -1) {
        switch (opt) {
        case 'n':
            memset(trainset_dir, '\0', FILENAME_LENGTH);
            snprintf(trainset_dir, FILENAME_LENGTH - 1, "%s", optarg);
            break;

        case 't':
            memset(testset_dir, '\0', FILENAME_LENGTH);
            snprintf(testset_dir, FILENAME_LENGTH - 1, "%s", optarg);
            break;

        case 's':
            train_size = atoi(optarg);
            break;

        case 'i':
            test_size = atoi(optarg);
            break;

        default:
            printf("Error: unknown argument");
            exit(-1);
            break;
        }
    }

    /*start up MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Barrier(MPI_COMM_WORLD);

    if (comm_size < 2) {
        printf("Should be more than 2 processes!\n");
        return -1;
    }

    if (comm_rank == 0) {
        printf("Get testset directory:%s\ntest size:%d\n", testset_dir, test_size);
    }

    if (comm_rank == 1) {
        printf("Get training set directory:%s\ntrain size:%d\n", trainset_dir, train_size);
    }

    test_record = (double*)malloc(sizeof(double) * DIMS);

    if (!test_record) {
        printf("Failed to alloc buffer to test_record.\n");
        MPI_Finalize();
        return -1;
    }

    //Generate and commit knn result type
    MPI_Type_contiguous(2, MPI_DOUBLE, &knn_result_type);
    MPI_Type_commit(&knn_result_type);

    /*Malloc buffer for each process*/
    if (comm_rank == 0) {
        testset = (double*)malloc(sizeof(double) * DIMS * test_size);

        if (testset == NULL) {
            MPI_Finalize();
            return -1;
        }

        test_size_read = readfile_to_buffer(testset_dir, testset, test_size);
        printf("Read %d test records.\n", test_size_read);
#ifdef DEBUG
        printf("The last record of testset:\n");
        int i = 0;

        for (; i < FEATURE_NUM + 1; i++) {
            if (testset[(test_size_read - 1) * DIMS + i] < 0.000001) {
                printf("0 ");
            } else {
                printf("%lf ", testset[(test_size_read - 1) * DIMS + i]);
            }
        }

        printf("\n");
#endif
    }

    if (comm_rank == 1) {
        trainset = (double*)malloc(sizeof(double) * DIMS * train_size);

        if (trainset == NULL) {
            MPI_Finalize();
            return -1;
        }

        train_size_read = readfile_to_buffer(trainset_dir, trainset, train_size);
        printf("Read %d train records.\n", train_size_read);
#ifdef DEBUG
        printf("The third record of trainset:\n");
        int i = 0;

        for (; i < FEATURE_NUM + 1; i++) {
            if (trainset[DIMS * 2 + i] < 0.000001) {
                printf("0 ");
            } else {
                printf("%lf ", trainset[DIMS * 2 + i]);
            }
        }

        printf("\n");
#endif
    }

    /*Create worker group with rank from 1 to n-1*/
    MPI_Group MPI_GROUP_WORLD;
    MPI_Group GROUP_WORKER;
    MPI_Comm  COMM_WORKER;
    int server_ranks[1] = {0};

    MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
    MPI_Group_excl(MPI_GROUP_WORLD, 1, server_ranks, &GROUP_WORKER);
    MPI_Comm_create(MPI_COMM_WORLD, GROUP_WORKER, &COMM_WORKER);

#ifdef DEBUG

    if (comm_rank > 0) {
        int new_rank_in_worker = 0;
        MPI_Comm_rank(COMM_WORKER, &new_rank_in_worker);
        printf("rank %d in COMM_WORLD, rank %d in COMM_WORKER\n", comm_rank, new_rank_in_worker);
    }

#endif

    /*Split training set, and scatter each of them to worker process*/
    if (comm_rank > 0) {
        int worker_size = 0;
        int batch_size = 0;
        int last_batch_size = 0;

        MPI_Comm_size(COMM_WORKER, &worker_size);
        batch_size = (train_size / worker_size) * DIMS;
        last_batch_size = (train_size % worker_size) * DIMS  + batch_size;

        send_displs = (int*)malloc(sizeof(int) * worker_size);

        if (!send_displs) {
            printf("Failed to alloc buffer to send_displs.\n");
            MPI_Finalize();
            return -1;
        }

        send_counts = (int*)malloc(sizeof(int) * worker_size);

        if (!send_counts) {
            printf("Failed to alloc buffer to send_counts.\n");
            MPI_Finalize();
            return -1;
        }

        train_batch = (double*)malloc(sizeof(double) * last_batch_size);

        if (!train_batch) {
            printf("Failed to alloc buffer to train_batch of each worker\n");
            MPI_Finalize();
            return -1;
        }

        for (i = 0; i < worker_size - 1; i++) {
            send_counts[i] = batch_size;
        }

        send_counts[worker_size - 1] = last_batch_size;

        for (i = 0; i < worker_size; i++) {
            send_displs[i] = i * batch_size;
        }

        MPI_Scatterv(trainset,
                     send_counts,
                     send_displs,
                     MPI_DOUBLE,
                     train_batch,
                     last_batch_size,
                     MPI_DOUBLE,
                     0,
                     COMM_WORKER);
#ifdef DEBUG
        printf("Scatter complete, the second record in rank %d is:\n", comm_rank);

        for (i = 0; i < FEATURE_NUM - 1; i++) {
            if (train_batch[DIMS + i] < 0.000001) {
                printf("0 ");
            } else {
                printf("%lf ", train_batch[DIMS + i]);
            }
        }

        printf("\n");
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /*Rank 0 is server, send a test record to worker group,
     * and recieve K-Nearest-Neighbour reply from each of them*/
    if (comm_rank == 0) {
        int sample_count = 0;
        int right_count = 0;
        MPI_Request* requests = NULL;
        MPI_Status* statuses = NULL;
        knn_result* results = NULL;

        requests = (MPI_Request*)malloc(sizeof(MPI_Request) * (comm_size - 1));

        if (!requests) {
            printf("Alloc buffer to MPI_Requests error.\n");
            MPI_Finalize();
            return -1;
        }

        statuses = (MPI_Status*)malloc(sizeof(MPI_Status) * (comm_size - 1));

        if (!statuses) {
            printf("Alloc buffer to MPI_Status error.\n");
            MPI_Finalize();
            return -1;
        }

        results = (knn_result*)malloc(sizeof(knn_result) * K_KNN * (comm_size - 1));

        if (!results) {
            printf("Alloc buffer to knn_result error.\n");
            MPI_Finalize();
            return -1;
        }

        for (i = 0; i < test_size; i++) {
            memcpy(test_record, &testset[i * DIMS], DIMS * sizeof(double));

            //send test record
            for (j = 0; j < comm_size - 1; j++) {
                MPI_Isend(test_record, //buffer address
                          DIMS, //count
                          MPI_DOUBLE,
                          j + 1, //dest rank, from 1 to comm_size - 1
                          i, //tag
                          MPI_COMM_WORLD,
                          &requests[j]);
            }

            MPI_Waitall(comm_size - 1, requests, statuses);

            //recieve result
            for (j = 0; j < comm_size - 1; j++) {
                MPI_Irecv(&(results[j * K_KNN]),
                          K_KNN, //count
                          knn_result_type,
                          MPI_ANY_SOURCE,
                          i, //tag
                          MPI_COMM_WORLD,
                          &requests[j]);
            }

            MPI_Waitall(comm_size - 1, requests, statuses);

            //calculate the results, and check right or wrong
            sample_count++;

            if (test_record[0] == knn_check_results_reduce(results, (comm_size - 1) * K_KNN)) {
                right_count++;
            }

            if ((sample_count % 1000) == 0) {
                printf("Test %d samples, correct rate:%lf\n", sample_count,
                       (double)right_count / (double)sample_count);
            }
        }

        if (requests) {
            free(requests);
        }

        if (statuses) {
            free(statuses);
        }

        if (results) {
            free(results);
        }
    }

    /*Rank 1 to n-1 are workers, recieve a test record from server,
     *and send back K-Nearest_Neighbour to server */
    if (comm_rank > 0) {
        knn_result* results = NULL;
        MPI_Status status;

        results = (knn_result*)malloc(sizeof(knn_result) * K_KNN);

        if (!results) {
            printf("Failed to alloc knn_result.\n");
            MPI_Finalize();
            return -1;
        }

        for (i = 0; i < test_size; i++) {
            //Get a test record from server
            MPI_Recv(test_record,
                     DIMS, //count
                     MPI_DOUBLE,
                     0, //source rank
                     i, //tag
                     MPI_COMM_WORLD,
                     &status);
            //Calculate the distances, and get the K nearest neighbours
            int train_batch_size = send_counts[comm_rank - 1];
            knn_get_neighbours_map(train_batch, train_batch_size, test_record, results);
            MPI_Send(results,
                     K_KNN, //count
                     knn_result_type,
                     0, //dest rank
                     i, //tag
                     MPI_COMM_WORLD);
        }

        if (results) {
            free(results);
        }

        if (send_counts) {
            free(send_counts);
        }

        if (send_displs) {
            free(send_displs);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (trainset) {
        free(trainset);
    }

    if (testset) {
        free(testset);
    }

    if (train_batch) {
        free(train_batch);
    }

    if (test_record) {
        free(test_record);
    }

    MPI_Finalize();
    return 0;
}

int readfile_to_buffer(char* filename, double* dataset, int datasize) {
    int fd = 0;
    int current_size = 0;
    char* buffer = NULL;
    int filesize_in_bytes = 0;
    struct stat filemetadata;

    int read_bytes = 0;

    int line_end = 1;

    if ((filename == NULL) || (dataset == NULL)) {
        return -1;
    }

    fd = open(filename, O_RDONLY);

    if (fd < 0) {
        return -1;
    }

    memset(&filemetadata, 0, sizeof(filemetadata));

    if (fstat(fd, &filemetadata) == -1) {
        printf("Can't get file stat.\n");
        return -1;
    }

    filesize_in_bytes = (int)filemetadata.st_size;
    buffer = (char*)malloc(filesize_in_bytes + 1);

    if (buffer == NULL) {
        printf("File too large, can't alloc buffer\n");
        return -1;
    }

    buffer[filesize_in_bytes] = '\0';

    printf("begin to read file. filesize = %d\n", filesize_in_bytes);

    if ((read_bytes = read(fd, buffer, filesize_in_bytes)) > 0) {
        int buffer_index = 0;
        int feature_index = 0;
        int dataset_index = 0;
        char feature[100];

        while (buffer_index < read_bytes) {
            if (line_end) {
                /*Get label first*/
                while ((feature[feature_index++] = buffer[buffer_index++]) != ';') {

                }

                feature[feature_index - 1] = '\0';
                dataset[current_size * DIMS + dataset_index++] = strtod(feature, NULL);
                feature_index = 0;
                line_end = 0;
            }

            /*Get features*/
            while (1) {
                feature[feature_index++] = buffer[buffer_index++];

                if ((feature[feature_index - 1] == ';') || (feature[feature_index - 1] == ' ')) {
                    break;
                }
            }

            if ((feature[feature_index -  1] == ';') && (line_end == 0)) {
                line_end = 1; //Get the end of line

                while ((buffer[buffer_index] == '\n') || (buffer[buffer_index] == '\r')) {
                    buffer_index++; //skip "enter"
                }

                feature[feature_index - 1] = '\0';
                dataset[current_size * DIMS + dataset_index++] = strtod(feature, NULL);
                current_size++;
                dataset_index = 0;
                feature_index = 0;
                continue;
            }

            feature[feature_index - 1] = '\0';
            dataset[current_size * DIMS + dataset_index++] = strtod(feature, NULL);
            feature_index = 0;
        }
    }

    if (buffer) {
        free(buffer);
    }

    close(fd);
    return current_size;
}

/*
 * Check the results, get K nearest neighbours from results, multi-way merge sorting
 * @param:
 * results: the nearest neighbours get from worker processes
 * neighbours: total count of 'results'
 * @return:
 * Label which is most frequent in K nearest neighbours
 * */
int knn_check_results_reduce(knn_result* results, int neighbours) {
    int i = 0;
    int j = 0;
    int worker_num = neighbours / K_KNN;
    int current_index[K_KNN];

    int label_statistics[LABEL_NUM];
    knn_result final_result[K_KNN];

    if (!results) {
        return -1;
    }

    memset(current_index, 0, sizeof(int) * K_KNN);
    memset(label_statistics, 0, sizeof(int) * LABEL_NUM);

    //Multiway merge sort
    for (i = 0; i < K_KNN; i++) {
        final_result[i].label = results[current_index[0]].label;
        final_result[i].distance = results[current_index[0]].distance;

        for (j = 1; j < worker_num; j++) {
            if (results[j * K_KNN + current_index[j]].distance < final_result[i].distance) {
                final_result[i].distance = results[j * K_KNN + current_index[j]].distance;
                final_result[i].label = results[j * K_KNN + current_index[j]].label;
                current_index[j]++;
                break;
            }
        }

        if (j == worker_num) { //result[current_index[0]] is the minimum value
            current_index[0]++;
        }
    }

    //Vote the result
    for (i = 0; i < K_KNN; i++) {
        label_statistics[(int)(final_result[i].label)]++; //label from 0 to 9
    }

    /*Return the most frequent label
     *What if two labels have the same frequency, this program return the first label it meets
     *Maybe better choices exist
     * */
    int max_frequent = 0;

    for (i = 0; i < LABEL_NUM; i++) {
        if (label_statistics[i] > max_frequent) {
            max_frequent = label_statistics[i];
        }
    }

    for (i = 0; i < LABEL_NUM; i++) {
        if (label_statistics[i] == max_frequent) {
            return i;
        }
    }

    return -1;
}

/*
 * Calculate distances between test record and each training batch, insert sorting
 * @param:
 * train_batch: the address of training batch
 * train_batch_size: the size of training batch
 * test_record: the test record get from server processes
 * results: the K nearest neighbours, it is an array, and memory is ready
 * @return:
 * 0,if succeed; -1,if failed
 * */
int knn_get_neighbours_map(double* train_batch, int train_batch_size, double* test_record,
                           knn_result* results) {
    int i = 0;
    int j = 0;
    int k = 0;
    int insert_pos = 0;
    int current_nearest_num = 0;
    double distance = 0.0;

    if ((!train_batch) || (!test_record) || (!results)) {
        return -1;
    }

    for (i = 0; i < (train_batch_size / DIMS); i++) {
        for (j = 1, distance = 0.0; j <= FEATURE_NUM; j++) {
            distance += POWER_OF_2(test_record[j] - train_batch[i * DIMS + j]);
        }

        //Insert into the right place of results
        if (distance < results[current_nearest_num].distance) { //Do insert
            //Get the right place
            for (insert_pos = current_nearest_num; insert_pos >= 0; insert_pos--) {
                if (distance > results[insert_pos].distance) {
                    break;
                }
            }

            insert_pos++;

            //Insert in results[insert_pos]
            for (k = current_nearest_num; k >= insert_pos; k--) {
                if (k < K_KNN - 1) { //if results[K_KNN - 1] exists, delete directly
                    results[k + 1].distance = results[k].distance;
                    results[k + 1].label = results[k].label;
                }
            }

            results[insert_pos].distance = distance;
            results[insert_pos].label = train_batch[i * DIMS];

            if (current_nearest_num < K_KNN - 1) {
                current_nearest_num++;
            }
        } else if (current_nearest_num < K_KNN - 1) { //Add to tail
            results[current_nearest_num].distance = distance;
            results[current_nearest_num].label = train_batch[i * DIMS];
            current_nearest_num++;
        }
    }

    return 0;
}
