#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <getopt.h>
#include <mpi.h>

int g_block_size = 1024*1024*100;
int g_iteration  = 720;
int g_server_num = 4;
int g_interval   = 0;

/*
 * Each server has 2 MPI Processes,
 * Odd ranks are worker processes,
 * Even ranks are server processes
*/

int main(int argc, char *argv[])
{
    int i = 0;
    int j = 0;
    int rank = 0;
    int size = 0;
    int tag = 1;
    int rz = 0;
    int *sendbuf = NULL;
    int *recvbuf = NULL;
    MPI_Status status;

    int opt = 0;
    int option_index = 0;
    static struct option long_options[] = {
        {"block_size", 1, 0, 'b'},
        {"interval", 1, 0, 't'},
        {"iteration", 1, 0, 'i'},
        {0, 0, 0, 0}
    };

    //resolve arguments
    while((opt = getopt_long(argc, argv, "b:t:i:", long_options, &option_index)) != -1){
        switch(opt){
            case 'b':
                g_block_size = atoi(optarg);
                printf("Set block_size = %d\n", g_block_size);
                break;
            case 't':
                g_interval = atoi(optarg);
                printf("Set g_interval = %d\n", g_interval);
                break;
            case 'i':
                g_iteration = atoi(optarg);
                printf("Set g_iteration = %d\n", g_iteration);
                break;
            default:
                printf("Error: unknown argument\n");
                break;
        }
    }

    sendbuf = (int *)memalign(64, sizeof(int) * g_block_size);
    recvbuf = (int *)memalign(64, sizeof(int) * g_block_size);

    if( !sendbuf || !recvbuf )
    {   
        printf("Initialization error,cannot alloc buffer.\n");
    }

    /*Assignment*/
    for(i = 0; i < g_block_size; i++)
    {   
        sendbuf[i] = i;
    }

    /*Start up MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD);

    /*Get Server Node Information, and send&recv rank*/
    g_server_num = size;
    int send_rank = 0;
    int recv_rank = 0;
    int iteration_count = 0;
    int server_num = g_server_num / 2;
    int worker_num = server_num;
    int source = 0;
    //int len = 0;

    if(rank % 2 == 0)  //parameter server
    {   
        for(iteration_count = 0; iteration_count < (g_iteration * worker_num); iteration_count++)
        {   
            MPI_Recv((void *)recvbuf, g_block_size,
                MPI_INT, MPI_ANY_SOURCE,
                tag, MPI_COMM_WORLD,
                &status);
            source = status.MPI_SOURCE;
            MPI_Get_count(&status, MPI_INT, &rz);
            if(rz != g_block_size)
            {   
                printf("rank[%d] to rank[%d],iteration[%d]:MPI_Recv Error!\n", source, rank, iteration_count / worker_num);
                exit(-1);
            }
            for(j = 0; j < g_block_size; j++) //check
            {   
                if(recvbuf[j] != j)
                {
                printf("Recv Data error.\n");
                exit(-1);
                }
            }

            printf("rank[%d] to rank[%d],iteration[%d]:MPI_Recv Success.\n", source, rank, iteration_count / worker_num);

            if(g_interval >= 1)      //computing
                sleep(g_interval);

            MPI_Send((void *)sendbuf, g_block_size,
                MPI_INT, source, tag, MPI_COMM_WORLD);

            printf("rank[%d] to rank[%d],iteration[%d]:MPI_Send Success.\n", rank, source, iteration_count / worker_num);
        }
    }
    else   //parameter worker
    {   
        for(iteration_count = 0; iteration_count < g_iteration; iteration_count++)
        {   
            for(i = 0; i < server_num; i++) //MPI_Send
            {  
               send_rank = rank;
               recv_rank = i * 2;
               MPI_Send((void *)sendbuf, g_block_size,
                   MPI_INT, recv_rank,
                   tag, MPI_COMM_WORLD);
               printf("rank[%d] to rank[%d],iteration[%d]:MPI_Send Success.\n", send_rank, recv_rank, iteration_count);

               send_rank = i * 2;
               recv_rank = rank;
               MPI_Recv((void *)recvbuf, g_block_size,
                    MPI_INT, send_rank,
                    tag, MPI_COMM_WORLD,
                    &status);
               MPI_Get_count(&status, MPI_INT, &rz);
               if(rz != g_block_size)
               {   
                   printf("rank[%d] to rank[%d],iteration[%d]:MPI_Recv Error!\n", send_rank, recv_rank, iteration_count);
                   exit(-1);
               }
               for(j = 0; j < g_block_size; j++)
               {   
                   if(recvbuf[j] != j)
                   {   
                       printf("Recv Data error.\n");
                       exit(-1);
                   }
               }
               printf("rank[%d] to rank[%d],iteration[%d]:MPI_Recv Success.\n", send_rank, recv_rank, iteration_count);
            }

            if(g_interval >= 1)           //computing
                sleep(g_interval);
        }
    }

    printf("Rank %d finished successfully.\n", rank);
	
    /*Cleanup MPI environment*/
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    if(sendbuf)
        free(sendbuf);
    if(recvbuf)
        free(recvbuf);
    return 0;
}
