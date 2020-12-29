#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "auxiliar/auxiliar.hpp"

extern "C" void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, const int *ldc);

int main(int argc, char* argv[])
{
  /**********************************************************
  *  Initial Setup
  **********************************************************/
  size_t N = 1024;
  double one = 1.0;

  double *A,*B,*C, *tmpA, *tmpB;
  MPI_Request request[2];

  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

 /**********************************************************
  *  Determining rank geometry < You need to improve this >
  **********************************************************/

  int p = sqrt(size);
  if (size != p*p) { printf("[Error] Number of processors must be a square integer.\n"); MPI_Abort(MPI_COMM_WORLD, -1); }

  /* Subquestion a: Use a Cartesian topology and determine the neighbors of each MPI rank */
  /************************************************************************************************/
  MPI_Comm cart_comm;
  int nums[2] = {0, 0};
  int periodic[2] = {true, true};
  
  MPI_Dims_create(size, 2, nums);
  MPI_Cart_create(MPI_COMM_WORLD, 2, nums, periodic, true, &cart_comm);

  MPI_Comm_rank(cart_comm, &rank);

  int coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, coords);

  int rankx = coords[0];
  int ranky = coords[1];

  int left, right, up, down;
  MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
  MPI_Cart_shift(cart_comm, 1, 1, &up, &down);
  /************************************************************************************************/

  // int ranky = rank/p;
  // int rankx = rank%p;

  // int up, down, left, right;
  // up = (rank - p + size)%size;
  // down = (rank + p)%size;
  // left = rank - 1 + (rank%p==0?p:0);
  // right = rank + 1 - (rank%p==p-1?p:0);

  /************************************************************************************************/

 /********************************************
  *   Initializing Matrices
  *******************************************/

  // Determining local matrix side size (n)
  const int n = N/p;

  // Allocating space for local matrices A, B, C
  A    = (double*) calloc (n*n, sizeof(double));
  B    = (double*) calloc (n*n, sizeof(double));
  C    = (double*) calloc (n*n, sizeof(double));

  // Allocating space for recieve buffers for A and B local matrices
  tmpA = (double*) calloc (n*n, sizeof(double));
  tmpB = (double*) calloc (n*n, sizeof(double));

  // Initializing values for input matrices A and B
  initializeMatrices(A, B, n, N, rankx, ranky);


  /********************************************
  *   Creating Datatypes
  *******************************************/
  /* Subquestion b: Create a custom MPI datatype to send and receive submatrices between ranks */
  /************************************************************************************************/
  MPI_Datatype subBlockType;
 
  MPI_Type_contiguous(n*n, MPI_DOUBLE, &subBlockType);

  MPI_Type_commit(&subBlockType);
  /************************************************************************************************/


 /**************************************************************
  *   Running Cannon's Matrix-Matrix  Multiplication Algorithm
  **************************************************************/

  if (rank == 0) printf("Running Matrix-Matrix Multiplication...\n");

  // Registering initial time. It's important to precede this with a barrier to make sure
  // all ranks are ready to start when we take the time.
  MPI_Barrier(MPI_COMM_WORLD);
  double execTime = -MPI_Wtime();

  dgemm_("N", "N", &n, &n, &n, &one, A, &n, B, &n, &one, C, &n);
  for(int step =1; step < p; step++)
  {
    /* Subquestion b: Create a custom MPI datatype to send and receive submatrices between ranks */
    /************************************************************************************************/
    MPI_Irecv(tmpA, 1, subBlockType, right, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(tmpB, 1, subBlockType, down, 1, MPI_COMM_WORLD, &request[1]);

    MPI_Send(A, 1, subBlockType, left, 0, MPI_COMM_WORLD);
    MPI_Send(B, 1, subBlockType, up, 1, MPI_COMM_WORLD);

    // Currently MPI_DOUBLE as Datatype, let's create a special datatype for communicating submatrices
    // MPI_Irecv(tmpA, n*n, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &request[0]);
    // MPI_Irecv(tmpB, n*n, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &request[1]);

    // MPI_Send(A, n*n, MPI_DOUBLE, left, 0, MPI_COMM_WORLD);
    // MPI_Send(B, n*n, MPI_DOUBLE, up, 1, MPI_COMM_WORLD);
    /************************************************************************************************/

    MPI_Waitall(2, request, MPI_STATUS_IGNORE);

    double* holdA= A; A = tmpA; tmpA = holdA;
    double* holdB= B; B = tmpB; tmpB = holdB;

    dgemm_("N", "N", &n, &n, &n, &one, A, &n, B, &n, &one, C, &n);
  }

  // Registering final time. It's important to precede this with a barrier to make sure all ranks have finished before we take the time.
  MPI_Barrier(MPI_COMM_WORLD);
  execTime += MPI_Wtime();

 /**************************************************************
  *   Verification Stage
  **************************************************************/

  verifyMatMulResults(C, n, N, rankx, ranky, execTime);
  free(A); free(B); free(C); free(tmpA); free(tmpB);

  MPI_Type_free(&subBlockType);
  return MPI_Finalize();
}
