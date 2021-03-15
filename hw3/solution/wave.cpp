#include "wave.h"

/*
Solution notes

Each thread is associated with a set of thread coordinates (t0,t1,t2) and the number of threads
per direction is p (so the total number of threads is p^3).

A thread lies at the boundary of a process if ti=0 and/or if ti=p-1, for i=0,1,2.

Threads at the boundary are associated with a unique send/pack id and a unique receive/unpack id.
Those ids are used to properly pack/unpack data and to match message tags when sending/receiving.



Each MPI process is assigned a cubic part of the domain. Thus, there are 6 faces where data needs 
to be exchanged. Each face has p^2 threads. This means that there are 6p^2 send/pack ids and 
6p^2 unpack/receive ids.

Logically, those ids correspond to a 3-dimensional array, with sizes [6][p][p]. They are stored in
1-dimensional arrays, so that 
[i][j][k] = i * p^2 + j * p + k 
i=0,1,...,5
j=0,1,...,p-1
k=0,1,...,p-1  


In the present solution, send ids correspond to receive ids as follows

send_id[0][j][k] = receive_id[1][j][k]
send_id[1][j][k] = receive_id[0][j][k]
send_id[2][j][k] = receive_id[3][j][k]
send_id[3][j][k] = receive_id[2][j][k]
send_id[4][j][k] = receive_id[5][j][k]
send_id[5][j][k] = receive_id[4][j][k]
for all j,k=0,...,p-1

*/



void WaveEquation::pack_all() {
  int array_of_sizes[3] = {N + 2, N + 2, N + 2};
  int nloc = N / threads_per_dim;
  int p = threads_per_dim;
  int tid = omp_get_thread_num();
  int t0, t1, t2;
  thread_coordinates(tid, threads_per_dim, t0, t1, t2);

  if (t0 == p - 1) {
    int pid = 0 * p * p + t1 * p + t2;
    int array_of_subsizes[3] = {1, nloc, nloc};
    int array_of_starts[3] = {N, 1 + t1 * nloc, 1 + t2 * nloc};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t0 == 0) {
    int pid = 1 * p * p + t1 * p + t2;
    int array_of_subsizes[3] = {1, nloc, nloc};
    int array_of_starts[3] = {1, 1 + t1 * nloc, 1 + t2 * nloc};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }

  if (t1 == p - 1) {
    int pid = 2 * p * p + t0 * p + t2;
    int array_of_subsizes[3] = {nloc, 1, nloc};
    int array_of_starts[3] = {1 + t0 * nloc, N, 1 + t2 * nloc};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t1 == 0) {
    int pid = 3 * p * p + t0 * p + t2;
    int array_of_subsizes[3] = {nloc, 1, nloc};
    int array_of_starts[3] = {1 + t0 * nloc, 1, 1 + t2 * nloc};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t2 == p - 1) {
    int pid = 4 * p * p + t0 * p + t1;
    int array_of_subsizes[3] = {nloc, nloc, 1};
    int array_of_starts[3] = {1 + t0 * nloc, 1 + t1 * nloc, N};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t2 == 0) {
    int pid = 5 * p * p + t0 * p + t1;
    int array_of_subsizes[3] = {nloc, nloc, 1};
    int array_of_starts[3] = {1 + t0 * nloc, 1 + t1 * nloc, 1};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
}

void WaveEquation::unpack_all() {
  int array_of_sizes[3] = {N + 2, N + 2, N + 2};
  int nloc = N / threads_per_dim;
  int p = threads_per_dim;
  int tid = omp_get_thread_num();
  int t0, t1, t2;
  thread_coordinates(tid, threads_per_dim, t0, t1, t2);

  if (t0 == 0) {
    int pid = 0 * p * p + t1 * p + t2;
    int array_of_subsizes[3] = {1, nloc, nloc};
    int array_of_starts[3] = {0, 1 + t1 * nloc, 1 + t2 * nloc};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes,array_of_starts);
  }
  if (t0 == p - 1) {
    int pid = 1 * p * p + t1 * p + t2;
    int array_of_subsizes[3] = {1, nloc, nloc};
    int array_of_starts[3] = {N + 1, 1 + t1 * nloc, 1 + t2 * nloc};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes,array_of_starts);
  }
  if (t1 == 0) {
    int pid = 2 * p * p + t0 * p + t2;
    int array_of_subsizes[3] = {nloc, 1, nloc};
    int array_of_starts[3] = {1 + t0 * nloc, 0, 1 + t2 * nloc};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes,array_of_starts);
  }
  if (t1 == p - 1) {
    int pid = 3 * p * p + t0 * p + t2;
    int array_of_subsizes[3] = {nloc, 1, nloc};
    int array_of_starts[3] = {1 + t0 * nloc, N + 1, 1 + t2 * nloc};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes,array_of_starts);
  }
  if (t2 == 0) {
    int pid = 4 * p * p + t0 * p + t1;
    int array_of_subsizes[3] = {nloc, nloc, 1};
    int array_of_starts[3] = {1 + t0 * nloc, 1 + t1 * nloc, 0};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes,array_of_starts);
  }
  if (t2 == p - 1) {
    int pid = 5 * p * p + t0 * p + t1;
    int array_of_subsizes[3] = {nloc, nloc, 1};
    int array_of_starts[3] = {1 + t0 * nloc, 1 + t1 * nloc, N + 1};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes,array_of_starts);
  }
}

void WaveEquation::run(double t_end) {
  t = 0;

  //as described above, we need 6p^2 packs and 6p^2 unpacks
  int n = N * N / threads_per_dim / threads_per_dim;
  for (int i = 0; i < 6 * threads_per_dim * threads_per_dim; i++) {
    pack[i] = new double[n];
    unpack[i] = new double[n];
  }

  double total_time = 0;

  int count = 0;
  MPI_Barrier(cart_comm);
  double t0 = MPI_Wtime();


//the only parallel region used startas here
#pragma omp parallel
  {

    int nloc = N / threads_per_dim;
    int p = threads_per_dim;
    int tid = omp_get_thread_num();
    int t0, t1, t2;
    thread_coordinates(tid, threads_per_dim, t0, t1, t2);
    int pid;

    //each thread finds the loop ranges it need only once, before starting the time iterations
    int i0_min = t0 * nloc;
    int i1_min = t1 * nloc;
    int i2_min = t2 * nloc;
    int i0_max = i0_min + nloc;
    int i1_max = i1_min + nloc;
    int i2_max = i2_min + nloc;

    do {
      
      //have only the master thread print screen output
      #pragma omp master
      {
        if (count % 10 == 0) {
          if (rank == 0)
            std::cout << count << " t=" << t << "\n";
            //Print(count);
        }
      }


      //Calling a function inside the parallel region means that each thread will enter the 
      //function 
      pack_all();


      //Here we declare an array of requests that is local to each thread.
      //This allows threads to send/receive messages independant of one another.
      std::vector<MPI_Request> local_request;


      if (t0 == 0) {
        local_request.resize(local_request.size() + 2);
        pid = 0 * p * p + t1 * p + t2;
        MPI_Irecv(unpack[pid], nloc * nloc, MPI_DOUBLE, rank_minus[0], pid,
                  cart_comm, &local_request[local_request.size() - 2]);

        pid = 1 * p * p + t1 * p + t2;
        MPI_Isend(pack[pid], nloc * nloc, MPI_DOUBLE, rank_minus[0], pid,
                  cart_comm, &local_request[local_request.size() - 1]);
      }
      if (t0 == p - 1) {
        local_request.resize(local_request.size() + 2);

        pid = 1 * p * p + t1 * p + t2;
        MPI_Irecv(unpack[pid], nloc * nloc, MPI_DOUBLE, rank_plus[0], pid,
                  cart_comm, &local_request[local_request.size() - 2]);

        pid = 0 * p * p + t1 * p + t2;
        MPI_Isend(pack[pid], nloc * nloc, MPI_DOUBLE, rank_plus[0], pid,
                  cart_comm, &local_request[local_request.size() - 1]);
      }
      if (t1 == 0) {
        local_request.resize(local_request.size() + 2);

        pid = 2 * p * p + t0 * p + t2;
        MPI_Irecv(unpack[pid], nloc * nloc, MPI_DOUBLE, rank_minus[1], pid,
                  cart_comm, &local_request[local_request.size() - 2]);

        pid = 3 * p * p + t0 * p + t2;
        MPI_Isend(pack[pid], nloc * nloc, MPI_DOUBLE, rank_minus[1], pid,
                  cart_comm, &local_request[local_request.size() - 1]);
      }
      if (t1 == p - 1) {
        local_request.resize(local_request.size() + 2);

        pid = 3 * p * p + t0 * p + t2;
        MPI_Irecv(unpack[pid], nloc * nloc, MPI_DOUBLE, rank_plus[1], pid,
                  cart_comm, &local_request[local_request.size() - 2]);

        pid = 2 * p * p + t0 * p + t2;
        MPI_Isend(pack[pid], nloc * nloc, MPI_DOUBLE, rank_plus[1], pid,
                  cart_comm, &local_request[local_request.size() - 1]);
      }
      if (t2 == 0) {
        local_request.resize(local_request.size() + 2);

        pid = 4 * p * p + t0 * p + t1;
        MPI_Irecv(unpack[pid], nloc * nloc, MPI_DOUBLE, rank_minus[2], pid,
                  cart_comm, &local_request[local_request.size() - 2]);

        pid = 5 * p * p + t0 * p + t1;
        MPI_Isend(pack[pid], nloc * nloc, MPI_DOUBLE, rank_minus[2], pid,
                  cart_comm, &local_request[local_request.size() - 1]);
      }
      if (t2 == p - 1) {
        local_request.resize(local_request.size() + 2);

        pid = 5 * p * p + t0 * p + t1;
        MPI_Irecv(unpack[pid], nloc * nloc, MPI_DOUBLE, rank_plus[2], pid,
                  cart_comm, &local_request[local_request.size() - 2]);

        pid = 4 * p * p + t0 * p + t1;
        MPI_Isend(pack[pid], nloc * nloc, MPI_DOUBLE, rank_plus[2], pid,
                  cart_comm, &local_request[local_request.size() - 1]);
      }


      //Now that all the send/receive requests are posted, we proceed with the computation of 
      //inner points, for all threads
      for (int i0 = 2 + i0_min; i0 < i0_max; i0++)
        for (int i1 = 2 + i1_min; i1 < i1_max; i1++)
          for (int i2 = 2 + i2_min; i2 < i2_max; i2++)
            UpdateGridPoint(i0, i1, i2);


      //Threads at the boundary will wait here for communication to complete, others will proceed
      //because their local copy of local_requests will be empty
      MPI_Waitall(local_request.size(), local_request.data(),
                  MPI_STATUSES_IGNORE);

      //Now that communication is complete, boundary threads can unpack data
      unpack_all();


      //The following commented out loop would not overlap communication and computation.
      //It is replaced by the for loop before the MPI_Wait call and by the for loops that follow.
       //for (int i0 = 1 + i0_min; i0 < i0_max +1 ; i0++)
       //for (int i1 = 1 + i1_min; i1 < i1_max +1 ; i1++)
       //for (int i2 = 1 + i2_min; i2 < i2_max +1 ; i2++)
       // UpdateGridPoint(i0, i1, i2);



      //Update all boundary points on the planes i0=i0_min+1 and i0=i0_max
      for (int i1 = 1 + i1_min; i1 < i1_max + 1; i1++)
      for (int i2 = 1 + i2_min; i2 < i2_max + 1; i2++)
      {
        UpdateGridPoint(1 + i0_min, i1, i2);
        UpdateGridPoint(    i0_max, i1, i2);
      }

      //Update all boundary points on the planes i1=i1_min+1 and i1=i1_max
      //Note that the i0 range starts at 2+i0_min and ends at i0_max, because the points 1+i0_min 
      //and i0_max were updated in the previous loop
      for (int i0 = 2 + i0_min; i0 < i0_max     ; i0++)
      for (int i2 = 1 + i2_min; i2 < i2_max + 1 ; i2++)
      {
        UpdateGridPoint(i0, 1 + i1_min, i2);
        UpdateGridPoint(i0,     i1_max, i2);
      }


      //Update all boundary points on the planes i2=i2_min+1 and i2=i2_max
      //Note that the i0 range starts at 2+i0_min and ends at i0_max, because the points 1+i0_min 
      //and i0_max were updated in the previous loop. The same is now true for the range of i1      
      for (int i0 = 2 + i0_min; i0 < i0_max     ; i0++)
      for (int i1 = 2 + i1_min; i1 < i1_max     ; i1++)
      {
        UpdateGridPoint(i0, i1, 1 + i2_min);
        UpdateGridPoint(i0, i1,     i2_max);
      }


      //Now we need to swap pointers. Before this is done, we make sure that all threads are done
      //with grid point updates (remember that the threads share memory, so if one of them swaps
      //the pointers, the others will update the wrong grid points, if they are still updating).
#pragma omp barrier

      //Here we make sure that only one thread swaps the pointers and updates the time
      //pragma omp master is also fine
#pragma omp single
      {
        std::swap(u_new, u);
        std::swap(u_new, u_old);
        t += dt;
        count++;
      }
    } while (t < t_end);
  }

  MPI_Barrier(cart_comm);
  total_time = MPI_Wtime() - t0;

  double total_time_max;
  MPI_Reduce(&total_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, 0,
             cart_comm);
  if (rank == 0) {
    std::cout << "Total time = " << total_time_max << "\n";
  }
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

  for (int i = 6 * threads_per_dim * threads_per_dim - 1; i >= 0; i--) {
    delete[] pack[i];
    delete[] unpack[i];
  }
}