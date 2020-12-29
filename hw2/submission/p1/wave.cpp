#include "wave.h"

/********************************************************************/ 
/* Subquestion a: change the following function and use a Cartesian */ 
/* topology to find coords[3], rank_plus[3] and rank_minus[3]       */
/********************************************************************/
void WaveEquation::FindCoordinates() {

  int nums[3] = {0, 0, 0};
  // int periodic[3] = {true, true, true};
  int periodic[3] = {false, false, false};

  MPI_Dims_create(size, 3, nums);
  MPI_Cart_create(MPI_COMM_WORLD, 3, nums, periodic, true, &cart_comm);

  MPI_Comm_rank(cart_comm, &rank);

  MPI_Cart_coords(cart_comm, rank, 3, coords);

  int left, right, bottom, top, front, back;
  MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
  MPI_Cart_shift(cart_comm, 1, 1, &bottom, &top);
  MPI_Cart_shift(cart_comm, 2, 1, &front, &back);

  rank_plus[0] = right;
  rank_plus[1] = top;
  rank_plus[2] = back;

  rank_minus[0] = left;
  rank_minus[1] = bottom;
  rank_minus[2] = front;

  // std::cout << "rank_plus: " << std::endl;
  // for(int i=0; i<3; i++) std::cout << rank_plus[i] << " ";
  // std::cout << "rank_minus: " << std::endl;
  // for(int i=0; i<3; i++) std::cout << rank_minus[i] << " ";
}

void WaveEquation::run(double t_end) {

  t = 0;

  /********************************************************************/ 
  /* Subquestion b: you need to define 12 custom datatypes.           */
  /* For sending data, six datatypes (one per face) are required.     */
  /* For receiving data, six more datatypes are required.             */
  /* You should use MPI_Type_create_subarray for those datatypes.     */
  /********************************************************************/

  /* Subquestion b: Create and commit custom datatypes here */
  /************************************************************************************************/
  MPI_Datatype SEND_FACE_PLUS [3];
  MPI_Datatype SEND_FACE_MINUS[3]; 

  MPI_Datatype RECV_FACE_PLUS [3];
  MPI_Datatype RECV_FACE_MINUS[3];

  int array_of_sizes[3] = {N + 2, N + 2, N + 2};
  int array_of_subsizes[3][3] = {{1, N, N}, {N, 1, N}, {N, N, 1}};
  int array_of_starts[12][3] = {{N, 1, 1}, {N+1, 1, 1}, {1, 1, 1}, {0, 1, 1},
                                {1, N, 1}, {1, N+1, 1}, {1, 1, 1}, {1, 0, 1},
                                {1, 1, N}, {1, 1, N+1}, {1, 1, 1}, {1, 1, 0}};

  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[0], array_of_starts[0], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_PLUS[0]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[0], array_of_starts[1], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_PLUS[0]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[0], array_of_starts[2], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_MINUS[0]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[0], array_of_starts[3], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_MINUS[0]);

  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[1], array_of_starts[4], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_PLUS[1]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[1], array_of_starts[5], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_PLUS[1]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[1], array_of_starts[6], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_MINUS[1]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[1], array_of_starts[7], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_MINUS[1]);

  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[2], array_of_starts[8], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_PLUS[2]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[2], array_of_starts[9], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_PLUS[2]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[2], array_of_starts[10], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_MINUS[2]);
  MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes[2], array_of_starts[11], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_MINUS[2]);

  for (int i=0; i<3; i++)
  {
    MPI_Type_commit(&SEND_FACE_PLUS[i]);
    MPI_Type_commit(&SEND_FACE_MINUS[i]);

    MPI_Type_commit(&RECV_FACE_PLUS[i]);
    MPI_Type_commit(&RECV_FACE_MINUS[i]);
  }

  /************************************************************************************************/

  int count = 0;
  do {
    if (count % 100 == 0) {
      if (rank == 0)
        std::cout << count << " t=" << t << "\n";
      Print(count);
    }

    MPI_Request request[12];

    /* Subquestion b: Replace the sends and receives with ones that correspond to custom datatypes*/
    /**********************************************************************************************/
    MPI_Irecv(u, 1, RECV_FACE_MINUS[0], rank_minus[0], 100, cart_comm, &request[0]);
    MPI_Isend(u, 1,  SEND_FACE_PLUS[0], rank_plus [0], 100, cart_comm, &request[1]);

    MPI_Irecv(u, 1,  RECV_FACE_PLUS[0], rank_plus [0], 101, cart_comm, &request[2]);
    MPI_Isend(u, 1, SEND_FACE_MINUS[0], rank_minus[0], 101, cart_comm, &request[3]);

    MPI_Irecv(u, 1, RECV_FACE_MINUS[1], rank_minus[1], 200, cart_comm, &request[4]);
    MPI_Isend(u, 1,  SEND_FACE_PLUS[1], rank_plus [1], 200, cart_comm, &request[5]);

    MPI_Irecv(u, 1,  RECV_FACE_PLUS[1], rank_plus [1], 201, cart_comm, &request[6]);
    MPI_Isend(u, 1, SEND_FACE_MINUS[1], rank_minus[1], 201, cart_comm, &request[7]);

    MPI_Irecv(u, 1, RECV_FACE_MINUS[2], rank_minus[2], 300, cart_comm, &request[8]);
    MPI_Isend(u, 1,  SEND_FACE_PLUS[2], rank_plus [2], 300, cart_comm, &request[9]);

    MPI_Irecv(u, 1,  RECV_FACE_PLUS[2], rank_plus [2], 301, cart_comm, &request[10]);
    MPI_Isend(u, 1, SEND_FACE_MINUS[2], rank_minus[2], 301, cart_comm, &request[11]);
    /**********************************************************************************************/

    // Wait for communication to finish
    MPI_Waitall(12, &request[0], MPI_STATUSES_IGNORE);

    for (int i0 = 1; i0 <= N; i0++)
      for (int i1 = 1; i1 <= N; i1++)
        for (int i2 = 1; i2 <= N; i2++)
          UpdateGridPoint(i0, i1, i2);

    double *temp = u_old;
    u_old = u;
    u = u_new;
    u_new = temp;
    t += dt;
    count++;
  } while (t < t_end);

  double s = 0;
  double Checksum = 0;
  for (int k = 1; k <= N; k++)
    for (int j = 1; j <= N; j++)
      for (int i = 1; i <= N; i++) {
        int m = k + j * (N + 2) + i * (N + 2) * (N + 2);
        s += u[m] * u[m];
      }

  MPI_Reduce(&s, &Checksum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
  if (rank == 0)
    std::cout << "Checksum = " << Checksum << "\n";

  /* Subquestion b: You should free the custom datatypes and the communicator here. */
  for (int i=0; i<3; i++)
  {
    MPI_Type_free(&SEND_FACE_PLUS[i]);
    MPI_Type_free(&SEND_FACE_MINUS[i]);

    MPI_Type_free(&RECV_FACE_PLUS[i]);
    MPI_Type_free(&RECV_FACE_MINUS[i]);
  }
  
  MPI_Comm_free(&cart_comm);
  
}
